import os
import numpy as np
import asyncio
import traceback
from rouge_score import rouge_scorer
from dataclasses import replace

from helm.common.hierarchical_logger import hlog, htrack_block
from nltk.tokenize import sent_tokenize
from helm.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from helm.benchmark.model_deployment_registry import get_model_deployment, ModelDeployment
from .utils_contamination import UtilsContamination
from transformers import AutoTokenizer

from .prompt_translations import TS_GUESSING_MULTICHOICE


class TSGuessingQuestionMultiChoiceContaminationEvaluator:
    """
    Implements a question-based multi-choice guessing test for contamination detection.
    Prompts are constructed with a randomly masked incorrect option. The model is asked
    to predict the content of the mask. This prediction is then compared against the
    original text of that specific masked (incorrect) option.
    Reference: https://aclanthology.org/2024.naacl-long.482/
    """

    DEFAULT_MODEL_MAX_CONTEXT_TOKENS = 4096
    SMALL_MODEL_CONTEXT_THRESHOLD = 1024
    MAX_OUTPUT_TOKENS = 100
    TOKENIZER_BUFFER = 30
    GENERATION_TEMPERATURE = 0.1

    def __init__(self):
        self.language: str = "en"
        self.alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self._check_nltk_punkt()
        self.model_max_length: int = self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS

    def _check_nltk_punkt(self):
        try:
            sent_tokenize("Test sentence for NLTK punkt check.")
        except LookupError:
            hlog("WARNING: NLTK 'punkt' package not found. Needed for sentence tokenization.")
            hlog("WARNING: Please download it by running: import nltk; nltk.download('punkt')")

    def evaluate(
        self,
        executor,
        benchmark_path: str,
        scenario_state,
        language: str,
        tokenizer_service: TokenizerService
    ) -> list:
        return asyncio.run(self._evaluate_async(
            executor,
            benchmark_path,
            scenario_state,
            language,
            tokenizer_service
        ))

    def _get_prompt_parts(self) -> tuple:
        if self.language in TS_GUESSING_MULTICHOICE:
            parts = TS_GUESSING_MULTICHOICE[self.language]
        else:
            hlog(f"WARNING: Language '{self.language}' not found in TS_GUESSING_MULTICHOICE. Falling back to English.")
            parts = TS_GUESSING_MULTICHOICE["en"]
        
        return (
            parts["instruction_fill_option"],
            parts["instruction_knowledge"],
            parts["instruction_rule"],
            parts["header_question"],
            parts["header_options"],
            parts["footer_reply"]
        )

    async def _evaluate_async(
        self,
        executor,
        benchmark_path: str,
        scenario_state,
        language: str,
        tokenizer_service: TokenizerService
    ) -> list:
        self.language = language.lower()
        with htrack_block("TS-Guessing (question-multichoice) contamination evaluation"):
            eval_data_name = os.path.basename(benchmark_path).split(":")[0]
            if not (
                hasattr(scenario_state, "adapter_spec")
                and hasattr(scenario_state.adapter_spec, "method")
                and scenario_state.adapter_spec.method == "multiple_choice_joint"
            ):
                hlog(f'The selected benchmark "{eval_data_name}" does not qualify for this contamination detection strategy.')
                return []

            model_deployment_name_from_spec = scenario_state.adapter_spec.model_deployment or scenario_state.adapter_spec.model
            if not model_deployment_name_from_spec:
                hlog("ERROR: Model identifier (model_deployment or model) not found in AdapterSpec. Cannot proceed.")
                return []

            primary_source_max_length_found = False
            try:
                model_deployment: ModelDeployment = get_model_deployment(model_deployment_name_from_spec)
                if model_deployment.max_sequence_length is not None:
                    self.model_max_length = model_deployment.max_sequence_length
                    primary_source_max_length_found = True
                    hlog(f"Using model_max_length from ModelDeployment.max_sequence_length: {self.model_max_length} for {model_deployment_name_from_spec}")
                elif model_deployment.max_request_length is not None:
                    self.model_max_length = model_deployment.max_request_length
                    primary_source_max_length_found = True
                    hlog(f"Using model_max_length from ModelDeployment.max_request_length: {self.model_max_length} for {model_deployment_name_from_spec}")
                
                if not primary_source_max_length_found:
                    hlog(f"WARNING: max_sequence_length and max_request_length not set in ModelDeployment for '{model_deployment_name_from_spec}'.")
                    raise ValueError("Relevant max length fields not in ModelDeployment")

            except (ValueError, KeyError) as e_model_reg:
                hlog(f"INFO: Could not get max length from ModelDeployment for '{model_deployment_name_from_spec}': {e_model_reg}. "
                     "Falling back to transformers.AutoTokenizer or default method.")
                temp_tokenizer_for_max_len: AutoTokenizer | None = None
                try:
                    hlog(f"Fallback: Loading transformers.AutoTokenizer for: {model_deployment_name_from_spec} to get model_max_length.")
                    temp_tokenizer_for_max_len = AutoTokenizer.from_pretrained(model_deployment_name_from_spec, trust_remote_code=True) 
                    tokenizer_max_len_attr = getattr(temp_tokenizer_for_max_len, 'model_max_length', self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS)

                    if not isinstance(tokenizer_max_len_attr, int) or tokenizer_max_len_attr <= 0 or tokenizer_max_len_attr > 2000000:
                        hlog(f"WARNING: Fallback tokenizer for {model_deployment_name_from_spec} reported an unusual max length ({tokenizer_max_len_attr}). "
                             f"Using default: {self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS}.")
                        self.model_max_length = self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS
                    else:
                        self.model_max_length = tokenizer_max_len_attr
                except Exception as e_hf_tokenizer:
                    hlog(f"ERROR: Fallback using transformers.AutoTokenizer for '{model_deployment_name_from_spec}' also failed: {e_hf_tokenizer}. "
                         f"Using default model_max_length: {self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS}.")
                    self.model_max_length = self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS
                finally:
                    if temp_tokenizer_for_max_len:
                        del temp_tokenizer_for_max_len 
                hlog(f"Using model_max_length (from fallback): {self.model_max_length} for {model_deployment_name_from_spec}")
            
            if self.model_max_length <= self.SMALL_MODEL_CONTEXT_THRESHOLD:
                hlog(f"WARNING: Model {model_deployment_name_from_spec} (effective context window {self.model_max_length} tokens) "
                     "may skip many instances if prompts are too long.")

            data_points = self._filter_data(scenario_state)
            hlog(f"Filtered to {len(data_points)} data points.")
            if not data_points: return []

            shuffled_data_points = [data_points[i] for i in np.random.permutation(len(data_points))]

            reference_texts_for_masked_slots = []
            masked_option_letters = []
            valid_request_states_for_execution = []
            skipped_instance_count = 0

            prompt_components = self._get_prompt_parts()

            generation_adapter_spec = replace(
                scenario_state.adapter_spec,
                method="generation",
                instructions="", 
                input_prefix="", 
                output_prefix="Answer: ", 
                input_suffix="", 
                output_suffix="",
                max_tokens=self.MAX_OUTPUT_TOKENS, 
                temperature=self.GENERATION_TEMPERATURE,
                stop_sequences=[], 
                num_outputs=1, 
                global_prefix="",
                global_suffix="",
                reference_prefix="", 
                reference_suffix="",
            )

            for data_point_item in shuffled_data_points:
                original_idx = data_point_item["original_request_state_index"]
                if original_idx >= len(scenario_state.request_states):
                    hlog(f"DEBUG: original_request_state_index {original_idx} out of bounds. Skipping.")
                    continue
                current_request_state = scenario_state.request_states[original_idx]
                try:
                    instruction_text, user_text, original_text_of_masked_option, wrong_letter = self._build_prompt(
                        data_point_item, prompt_components
                    )
                    if instruction_text == "failed": continue
                    
                    combined_prompt = f"{instruction_text}\n\n{user_text}"
                    
                    try:
                        tokenization_request = TokenizationRequest(text=combined_prompt, tokenizer=model_deployment_name_from_spec)
                        tokenization_result: TokenizationRequestResult = tokenizer_service.tokenize(tokenization_request)
                        num_prompt_tokens = len(tokenization_result.tokens)
                    except Exception as e_tok_service:
                        hlog(f"ERROR: TokenizerService failed for instance {original_idx}. Skipping. Error: {e_tok_service} - {traceback.format_exc()}")
                        skipped_instance_count += 1
                        continue
                        
                    max_allowable_prompt_tokens = self.model_max_length - self.MAX_OUTPUT_TOKENS - self.TOKENIZER_BUFFER
                    if num_prompt_tokens > max_allowable_prompt_tokens:
                        skipped_instance_count += 1
                        continue
                        
                    new_input = replace(current_request_state.instance.input, text=combined_prompt)
                    new_instance = replace(current_request_state.instance, input=new_input, references=[])
                    new_request = replace(
                        current_request_state.request, prompt=combined_prompt,
                        max_tokens=self.MAX_OUTPUT_TOKENS, temperature=self.GENERATION_TEMPERATURE
                    )
                    prepared_rs = replace(current_request_state, instance=new_instance, request=new_request, result=None)
                    if hasattr(prepared_rs, "output_mapping"): prepared_rs = replace(prepared_rs, output_mapping=None)
                    if hasattr(prepared_rs, "reference_index"): prepared_rs = replace(prepared_rs, reference_index=None)
                    
                    valid_request_states_for_execution.append(prepared_rs)
                    reference_texts_for_masked_slots.append(original_text_of_masked_option)
                    masked_option_letters.append(wrong_letter)
                except Exception as e:
                    hlog(f"Error preparing instance {original_idx}: {e} - {traceback.format_exc()}")

            if skipped_instance_count > 0:
                hlog(f"INFO: Skipped {skipped_instance_count} instances due to length/tokenization.")
            if not valid_request_states_for_execution:
                hlog("No instances prepared for execution.")
                return []
                
            hlog(f"Sending {len(valid_request_states_for_execution)} requests.")
            execution_scenario_state = replace(
                scenario_state, adapter_spec=generation_adapter_spec, request_states=valid_request_states_for_execution
            )
            try:
                response_scenario_state = await self._query_model(execution_scenario_state, executor)
            except Exception as e:
                hlog(f"Error during model query: {e}")
                return []

            processed_instance_results = []
            for i, response_state in enumerate(response_scenario_state.request_states):
                try:
                    if response_state.result and response_state.result.completions:
                        response_text = response_state.result.completions[0].text.strip()
                        processed_response = self._process_response(response_text, masked_option_letters[i]) 
                        instance_id = getattr(valid_request_states_for_execution[i].instance, "id", f"proc_inst_{i}")
                        processed_instance_results.append({
                            "id": instance_id,
                            "answer_to_compare": reference_texts_for_masked_slots[i].lower(),
                            "model_prediction": processed_response.lower()
                        })
                except Exception as e:
                    hlog(f"Error processing result for prepared_instance_index {i}: {e}")

            if not processed_instance_results:
                hlog("No valid results after model query.")
                return []

            gold_references = [res["answer_to_compare"] for res in processed_instance_results]
            model_generations = [res["model_prediction"] for res in processed_instance_results]
            
            exact_match_score = 0.0
            if model_generations:
                exact_match_score = sum(gen == gold for gen, gold in zip(model_generations, gold_references)) / len(model_generations)

            calculated_metrics = {
                "exact_match": exact_match_score,
                "rouge_l_f1": 0.0
            }

            if model_generations:
                try:
                    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=(self.language == "en"))
                    rouge_scores_f1 = [
                        scorer.score(str(model_generations[i_score]), str(gold_references[i_score]))["rougeLsum"].fmeasure
                        if gold_references[i_score] and model_generations[i_score]
                        else 0.0
                        for i_score in range(len(model_generations))
                    ]
                    if rouge_scores_f1: calculated_metrics["rouge_l_f1"] = np.mean(rouge_scores_f1)
                except Exception as e:
                    hlog(f"ROUGE Scorer error: {e}")

            final_helm_stats = []
            for metric_name, metric_value in calculated_metrics.items():
                metric_value = round(metric_value, 2)
                final_helm_stats.append(
                    {
                        "name": {"name": f"contamination (ts guessing multichoice {metric_name})", "split": "test"},
                        "count": 1, 
                        "sum": metric_value, 
                        "sum_squared": metric_value**2,
                        "min": metric_value, 
                        "max": metric_value, 
                        "mean": metric_value,
                        "variance": 0.0, 
                        "stddev": 0.0 
                    }
                )
            hlog(f"Evaluation completed for language '{self.language}'. Processed {len(processed_instance_results)}. Skipped {skipped_instance_count}.")
            return final_helm_stats

    def _filter_data(self, scenario_state):
        data_points = []
        for i, request_state_item in enumerate(scenario_state.request_states):
            try:
                question_text = UtilsContamination.get_question_text(request_state_item.instance)
                choices_list = UtilsContamination.get_choices(request_state_item.instance)
                true_correct_answer_idx = UtilsContamination.get_answer_index(request_state_item.instance)

                if question_text and choices_list and isinstance(true_correct_answer_idx, int) and 0 <= true_correct_answer_idx < len(choices_list):
                    data_points.append({
                        "id": getattr(request_state_item.instance, "id", f"instance_{i}"),
                        "text": question_text,
                        "choices": choices_list,
                        "true_correct_answer_index": true_correct_answer_idx,
                        "original_request_state_index": i 
                    })
            except Exception as e:
                hlog(f"Error filtering data point at index {i}: {e}")
        return data_points

    def _build_prompt(self, example_item, prompt_components_tuple):
        try:
            (part_instr_fill, part_instr_knowledge, part_instr_rule,
             part_header_q, part_header_opts, part_footer_reply) = prompt_components_tuple

            original_question_text = example_item.get("text", "")
            choices_list = example_item.get("choices", [])
            true_correct_answer_idx = example_item.get("true_correct_answer_index", -1)

            if not (original_question_text and choices_list and
                    isinstance(true_correct_answer_idx, int) and 0 <= true_correct_answer_idx < len(choices_list)):
                hlog(f"DEBUG: Failed _build_prompt due to invalid example_item data for ID: {example_item.get('id', 'Unknown')}")
                return "failed", "", "", ""

            incorrect_option_indices = [i for i in range(len(choices_list)) if i != true_correct_answer_idx]
            if not incorrect_option_indices:
                hlog(f"DEBUG: No incorrect options to mask for instance {example_item.get('id', 'Unknown')}.")
                return "failed", "", "", ""

            masked_choice_idx = np.random.choice(incorrect_option_indices)
            original_text_of_masked_option = str(choices_list[masked_choice_idx])
            masked_option_letter = self.alphabet[masked_choice_idx % len(self.alphabet)]

            instruction_text_segment = f"{part_instr_fill} {masked_option_letter}.\n{part_instr_knowledge}\n\n{part_instr_rule}"

            options_lines_segment = ""
            for i, choice_text_item in enumerate(choices_list):
                if i >= len(self.alphabet): continue 
                letter = self.alphabet[i]

                option_content = "[MASK]" if i == masked_choice_idx else f"[{str(choice_text_item)}]"
                options_lines_segment += f"\n{letter}: {option_content}"
            
            user_text_segment = f"{part_header_q} {original_question_text}\n{part_header_opts}{options_lines_segment}\n\n{part_footer_reply}"

            return instruction_text_segment, user_text_segment, original_text_of_masked_option, masked_option_letter
        except Exception as e:
            hlog(f"EXCEPTION: Critical error in _build_prompt for example {example_item.get('id', 'Unknown ID')}: {e} - {traceback.format_exc()}")
            return "failed", "", "", ""

    def _process_response(self, response_text_input, masked_option_letter_param):
        try:
            if not response_text_input: return ""

            processed_text = str(response_text_input)

            option_prefix_to_strip = masked_option_letter_param + ":"
            if processed_text.startswith(option_prefix_to_strip):
                processed_text = processed_text[len(option_prefix_to_strip):].strip()
            
            try:
                sentences = sent_tokenize(processed_text)
                if sentences: processed_text = sentences[0]
            except LookupError:
                pass 
            except Exception:
                pass 

            processed_text = processed_text.replace("[MASK]", "").strip()

            if processed_text.startswith("[") and processed_text.endswith("]"):
                processed_text = processed_text[1:-1].strip()
            
            return processed_text
        except Exception as e:
            hlog(f"Error processing response: '{response_text_input[:50]}...': {e}")
            return ""

    async def _query_model(self, scenario_state, executor):
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, executor.execute, scenario_state)
        except Exception as e:
            hlog(f"Model query failed: {e}")
            raise e