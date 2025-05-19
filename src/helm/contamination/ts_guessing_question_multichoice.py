import os
import numpy as np
import asyncio
import traceback
from rouge_score import rouge_scorer
from dataclasses import replace
from typing import List, Dict, Any, Tuple

from helm.common.hierarchical_logger import hlog, htrack_block
from nltk.tokenize import sent_tokenize
from helm.common.tokenization_request import TokenizationRequest
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from .utils_contamination import UtilsContamination

class TSGuessingQuestionMultiChoiceContaminationEvaluator:
    """
    Implements a question-based multi-choice guessing test for contamination detection.
    Prompts are constructed with a randomly masked incorrect option. The model is asked
    to predict the content of the mask. This prediction is then compared against the
    original text of that specific masked (incorrect) option.
    Reference: https://aclanthology.org/2024.naacl-long.482/
    """

    STRATEGY_NAME: str = "ts_guessing_multichoice"
    STRATEGY_DISPLAY_NAME: str = "TS-Guessing (question-multichoice)"


    SMALL_MODEL_CONTEXT_THRESHOLD: int = 1024
    MAX_OUTPUT_TOKENS: int = 100
    TOKENIZER_BUFFER: int = 30
    GENERATION_TEMPERATURE: float = 0.1

    def __init__(self):
        self.language: str = "en"
        self.alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self._check_nltk_punkt()

    def _check_nltk_punkt(self):
        try:
            sent_tokenize("Test sentence for NLTK punkt check.")
        except LookupError:
            hlog("UTIL WARNING: NLTK 'punkt' package not found. Needed for sentence tokenization in _process_response.")
            hlog("UTIL WARNING: Please download it by running: import nltk; nltk.download('punkt')")
        except Exception as e:
            hlog(f"UTIL WARNING: NLTK punkt check failed with an unexpected error: {e}")


    def evaluate(
        self,
        executor,
        benchmark_path: str,
        scenario_state,
        language: str,
        tokenizer_service: TokenizerService
    ) -> List[Dict[str, Any]]:
        return asyncio.run(self._evaluate_async(
            executor,
            benchmark_path,
            scenario_state,
            language,
            tokenizer_service
        ))

    async def _evaluate_async(
        self,
        executor,
        benchmark_path: str,
        scenario_state,
        language: str,
        tokenizer_service: TokenizerService
    ) -> List[Dict[str, Any]]:
        self.language = language.lower().split('_')[0]

        with htrack_block(f"{self.STRATEGY_DISPLAY_NAME} contamination evaluation for language '{self.language}'"):
            eval_data_name = os.path.basename(benchmark_path).split(":")[0]
            if not (
                hasattr(scenario_state, "adapter_spec")
                and hasattr(scenario_state.adapter_spec, "method")
                and scenario_state.adapter_spec.method == "multiple_choice_joint"
            ):
                hlog(f'STRATEGY INFO: The selected benchmark "{eval_data_name}" does not qualify for this contamination detection strategy (requires multiple_choice_joint).')
                return []

            model_deployment_name_from_spec = scenario_state.adapter_spec.model_deployment or scenario_state.adapter_spec.model
            if not model_deployment_name_from_spec:
                hlog("STRATEGY ERROR: Model identifier (model_deployment or model) not found in AdapterSpec. Cannot proceed.")
                return []

            model_max_length = UtilsContamination.determine_model_max_length(model_deployment_name_from_spec)
            hlog(f"STRATEGY INFO: Effective model_max_length for {model_deployment_name_from_spec}: {model_max_length}")

            if model_max_length <= self.SMALL_MODEL_CONTEXT_THRESHOLD:
                hlog(f"STRATEGY WARNING: Model {model_deployment_name_from_spec} (context window {model_max_length} tokens) "
                     "may skip many instances if prompts are too long.")

            data_points = self._filter_data(scenario_state)
            hlog(f"STRATEGY INFO: Filtered to {len(data_points)} data points for evaluation.")
            if not data_points:
                hlog("STRATEGY INFO: No data points available after filtering.")
                return []

            shuffled_data_points = [data_points[i] for i in np.random.permutation(len(data_points))]

            reference_texts_for_masked_slots: List[str] = []
            masked_option_letters: List[str] = []
            valid_request_states_for_execution = []
            skipped_instance_count: int = 0

            prompt_components = UtilsContamination.get_prompt_fragments(self.STRATEGY_NAME, self.language)
            if not prompt_components:
                hlog(f"STRATEGY ERROR: Could not load prompt components for strategy '{self.STRATEGY_NAME}' and language '{self.language}'.")
                return []

            generation_params = {
                "max_tokens": self.MAX_OUTPUT_TOKENS,
                "temperature": self.GENERATION_TEMPERATURE,
                "stop_sequences": [], 
                "num_outputs": 1,
                "output_prefix": "Answer: ",
                "instructions": "", 
                "input_prefix": "", 
                "input_suffix": "", 
                "output_suffix": "",
                "global_prefix": "", 
                "global_suffix": "", 
                "reference_prefix": "", 
                "reference_suffix": ""
            }
            generation_adapter_spec = UtilsContamination.create_generation_adapter_spec(
                scenario_state.adapter_spec, generation_params
            )

            max_allowable_prompt_tokens = model_max_length - self.MAX_OUTPUT_TOKENS - self.TOKENIZER_BUFFER

            for data_point_item in shuffled_data_points:
                original_idx = data_point_item["original_request_state_index"]
                if not (0 <= original_idx < len(scenario_state.request_states)):
                    hlog(f"STRATEGY DEBUG: original_request_state_index {original_idx} out of bounds. Skipping.")
                    skipped_instance_count += 1
                    continue
                
                current_request_state = scenario_state.request_states[original_idx]
                
                try:
                    instruction_text, user_text, original_text_of_masked_option, wrong_letter = self._build_prompt(
                        data_point_item, prompt_components
                    )
                    if instruction_text == "failed":
                        skipped_instance_count += 1
                        continue
                    
                    combined_prompt = f"{instruction_text}\n\n{user_text}"
                    
                    is_valid_len, num_prompt_tokens = UtilsContamination.check_prompt_length(
                        combined_prompt, model_deployment_name_from_spec, tokenizer_service, max_allowable_prompt_tokens
                    )

                    if not is_valid_len:
                        hlog(f"STRATEGY DEBUG: Instance {original_idx} skipped. Prompt too long ({num_prompt_tokens} > {max_allowable_prompt_tokens}).")
                        skipped_instance_count += 1
                        continue
                        
                    new_input = replace(current_request_state.instance.input, text=combined_prompt)
                    new_instance = replace(current_request_state.instance, input=new_input, references=[]) 
                    
                    new_request = replace(
                        current_request_state.request,
                        prompt=combined_prompt,
                        max_tokens=self.MAX_OUTPUT_TOKENS, 
                        temperature=self.GENERATION_TEMPERATURE,
                        stop_sequences=[]
                    )
                    
                    prepared_rs = replace(
                        current_request_state,
                        instance=new_instance,
                        request=new_request,
                        result=None
                    )

                    if hasattr(prepared_rs, "output_mapping"): prepared_rs = replace(prepared_rs, output_mapping=None)
                    if hasattr(prepared_rs, "reference_index"): prepared_rs = replace(prepared_rs, reference_index=None)
                    
                    valid_request_states_for_execution.append(prepared_rs)
                    reference_texts_for_masked_slots.append(original_text_of_masked_option)
                    masked_option_letters.append(wrong_letter)

                except Exception as e:
                    hlog(f"STRATEGY ERROR: Error preparing instance {original_idx} for {data_point_item.get('id', 'Unknown')}: {e}\n{traceback.format_exc()}")
                    skipped_instance_count += 1

            if skipped_instance_count > 0:
                hlog(f"STRATEGY INFO: Skipped {skipped_instance_count} instances due to length, tokenization, or prompt building errors.")
            if not valid_request_states_for_execution:
                hlog("STRATEGY INFO: No instances prepared for execution after processing all data points.")
                return []
                
            hlog(f"STRATEGY INFO: Sending {len(valid_request_states_for_execution)} requests for model generation.")
            execution_scenario_state = replace(
                scenario_state, adapter_spec=generation_adapter_spec, request_states=valid_request_states_for_execution
            )
            
            try:
                response_scenario_state = await self._query_model(execution_scenario_state, executor)
            except Exception as e:
                hlog(f"STRATEGY CRITICAL: Error during model query phase: {e}\n{traceback.format_exc()}")
                return []

            processed_instance_results: List[Dict[str, str]] = []
            for i, response_state in enumerate(response_scenario_state.request_states):
                try:
                    if response_state.result and response_state.result.completions and response_state.result.completions[0].text:
                        response_text = response_state.result.completions[0].text.strip()
                        processed_response = self._process_response(response_text, masked_option_letters[i])
                        
                        instance_id = getattr(valid_request_states_for_execution[i].instance, "id", f"proc_inst_{i}")
                        processed_instance_results.append({
                            "id": instance_id,
                            "answer_to_compare": reference_texts_for_masked_slots[i].lower(),
                            "model_prediction": processed_response.lower()
                        })
                    else:
                        hlog(f"STRATEGY WARNING: No valid completion found for prepared instance index {i}.")
                except Exception as e:
                    hlog(f"STRATEGY ERROR: Error processing result for prepared instance index {i}: {e}\n{traceback.format_exc()}")

            if not processed_instance_results:
                hlog("STRATEGY INFO: No valid results obtained after model query and processing.")
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
                        scorer.score(str(model_gen), str(gold_ref))["rougeLsum"].fmeasure
                        if gold_ref and model_gen
                        else 0.0
                        for model_gen, gold_ref in zip(model_generations, gold_references)
                    ]
                    if rouge_scores_f1: calculated_metrics["rouge_l_f1"] = np.mean(rouge_scores_f1)
                except Exception as e:
                    hlog(f"STRATEGY ERROR: ROUGE Scorer calculation failed: {e}\n{traceback.format_exc()}")

            strategy_metric_prefix = f"contamination ({self.STRATEGY_NAME}"
            final_helm_stats = UtilsContamination.format_helm_stats(
                calculated_metrics,
                strategy_metric_prefix,
                split="test"
            )
            
            hlog(f"STRATEGY INFO: Evaluation completed for language '{self.language}'. "
                 f"Processed {len(processed_instance_results)} instances. "
                 f"Final Exact Match: {calculated_metrics['exact_match']:.2f}, "
                 f"ROUGE-L F1: {calculated_metrics['rouge_l_f1']:.2f}")
            return final_helm_stats

    def _filter_data(self, scenario_state) -> List[Dict[str, Any]]:
        data_points: List[Dict[str, Any]] = []
        for i, request_state_item in enumerate(scenario_state.request_states):
            try:
                question_text = UtilsContamination.get_question_text(request_state_item.instance)
                choices_list = UtilsContamination.get_choices(request_state_item.instance)
                true_correct_answer_idx = UtilsContamination.get_answer_index(request_state_item.instance)

                if (question_text and question_text != "Unknown question or context" and
                    choices_list and len(choices_list) > 1 and
                    isinstance(true_correct_answer_idx, int) and 
                    0 <= true_correct_answer_idx < len(choices_list)):
                    
                    data_points.append({
                        "id": getattr(request_state_item.instance, "id", f"instance_{i}"),
                        "text": question_text,
                        "choices": choices_list,
                        "true_correct_answer_index": true_correct_answer_idx,
                        "original_request_state_index": i
                    })
                else:
                    hlog(f"STRATEGY DEBUG: Skipping instance {i} during filtering due to missing/invalid data "
                         f"(Q: {'OK' if question_text else 'Fail'}, C: {len(choices_list) if choices_list else 0}, A_idx: {true_correct_answer_idx}).")
            except Exception as e:
                hlog(f"STRATEGY ERROR: Error filtering data point at index {i}: {e}\n{traceback.format_exc()}")
        return data_points

    def _build_prompt(self, example_item: Dict[str, Any], prompt_components: Dict[str, str]) -> Tuple[str, str, str, str]:
        try:
            part_instr_fill = prompt_components.get("instruction_fill_option", "Fill [MASK] in option")
            part_instr_knowledge = prompt_components.get("instruction_knowledge", "Use knowledge.")
            part_instr_rule = prompt_components.get("instruction_rule", "Provide only [MASK] text.")
            part_header_q = prompt_components.get("header_question", "Question:")
            part_header_opts = prompt_components.get("header_options", "Options:")
            part_footer_reply = prompt_components.get("footer_reply", "The text for [MASK] is:")

            original_question_text = example_item.get("text", "")
            choices_list = example_item.get("choices", [])
            true_correct_answer_idx = example_item.get("true_correct_answer_index", -1)

            if not (original_question_text and choices_list and len(choices_list) > 1 and
                    isinstance(true_correct_answer_idx, int) and 0 <= true_correct_answer_idx < len(choices_list)):
                hlog(f"STRATEGY DEBUG: _build_prompt failed - invalid input data for ID: {example_item.get('id', 'Unknown')}")
                return "failed", "", "", ""

            incorrect_option_indices = [i for i in range(len(choices_list)) if i != true_correct_answer_idx]
            if not incorrect_option_indices:
                hlog(f"STRATEGY DEBUG: No incorrect options to mask for instance {example_item.get('id', 'Unknown')}. This shouldn't happen if choices > 1.")
                return "failed", "", "", ""

            masked_choice_idx = np.random.choice(incorrect_option_indices)
            original_text_of_masked_option = str(choices_list[masked_choice_idx])
            
            masked_option_letter = self.alphabet[masked_choice_idx % len(self.alphabet)] 

            instruction_text_segment = f"{part_instr_fill} {masked_option_letter}.\n{part_instr_knowledge}\n\n{part_instr_rule}"

            options_lines_segment = ""
            for i, choice_text_item in enumerate(choices_list):
                if i >= len(self.alphabet): 
                    hlog(f"STRATEGY WARNING: Number of choices ({len(choices_list)}) exceeds alphabet size ({len(self.alphabet)}) for instance {example_item.get('id', 'Unknown')}. Truncating options display.")
                    break 
                letter = self.alphabet[i]
                option_content = "[MASK]" if i == masked_choice_idx else f"[{str(choice_text_item)}]"
                options_lines_segment += f"\n{letter}: {option_content}"
            
            user_text_segment = f"{part_header_q} {original_question_text}\n{part_header_opts}{options_lines_segment}\n\n{part_footer_reply}"

            return instruction_text_segment, user_text_segment, original_text_of_masked_option, masked_option_letter
        except Exception as e:
            hlog(f"STRATEGY EXCEPTION: Critical error in _build_prompt for example {example_item.get('id', 'Unknown ID')}: {e}\n{traceback.format_exc()}")
            return "failed", "", "", ""

    def _process_response(self, response_text_input: str, masked_option_letter_param: str) -> str:
        try:
            if not response_text_input: return ""
            processed_text = str(response_text_input).strip()

            option_prefix_to_strip_colon = f"{masked_option_letter_param}:"
            option_prefix_to_strip_dot = f"{masked_option_letter_param}."
            
            if processed_text.upper().startswith(option_prefix_to_strip_colon.upper()):
                processed_text = processed_text[len(option_prefix_to_strip_colon):].strip()
            elif processed_text.upper().startswith(option_prefix_to_strip_dot.upper()):
                processed_text = processed_text[len(option_prefix_to_strip_dot):].strip()
            
            try:
                if 'nltk' in globals() and hasattr(globals()['nltk'], 'download') and sent_tokenize:
                     sentences = sent_tokenize(processed_text)
                     if sentences: processed_text = sentences[0]
            except LookupError:
                pass 
            except Exception as e_sent_tok:
                hlog(f"STRATEGY DEBUG: sent_tokenize failed during response processing: {e_sent_tok}")

            processed_text = processed_text.replace("[MASK]", "").strip()

            if processed_text.startswith("[") and processed_text.endswith("]"):
                processed_text = processed_text[1:-1].strip()
            
            if (processed_text.startswith('"') and processed_text.endswith('"')) or \
               (processed_text.startswith("'") and processed_text.endswith("'")):
                processed_text = processed_text[1:-1].strip()

            return processed_text
        except Exception as e:
            hlog(f"STRATEGY ERROR: Error processing model response: '{response_text_input[:50]}...': {e}\n{traceback.format_exc()}")
            return ""


    async def _query_model(self, scenario_state, executor) -> Any:
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, executor.execute, scenario_state)
        except Exception as e:
            hlog(f"STRATEGY CRITICAL: Model query execution failed: {e}\n{traceback.format_exc()}")
            raise