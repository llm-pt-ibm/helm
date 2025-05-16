import os
import numpy as np
import asyncio
import traceback
from googletrans import Translator
from rouge_score import rouge_scorer
from dataclasses import replace

from helm.common.hierarchical_logger import hlog, htrack_block
from nltk.tokenize import sent_tokenize
from helm.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from helm.benchmark.window_services.tokenizer_service import TokenizerService

from .utils_contamination import UtilsContamination
from transformers import AutoTokenizer

class TSGuessingQuestionMultiChoiceContaminationEvaluator:
    """
    Implements a question-based multi-choice guessing test for contamination detection.
    Prompts are constructed with a randomly masked incorrect option. The model is asked
    to predict the content of the mask. This prediction is then compared against the
    original text of that specific masked (incorrect) option.
    Uses HELM's TokenizerService for token counting and attempts to get max context
    length from HELM's model client.
    https://aclanthology.org/2024.naacl-long.482/
    """

    DEFAULT_MODEL_MAX_CONTEXT_TOKENS = 4096
    SMALL_MODEL_CONTEXT_THRESHOLD = 1024
    MAX_OUTPUT_TOKENS = 100
    TOKENIZER_BUFFER = 30
    GENERATION_TEMPERATURE = 0.1

    def __init__(self):
        self.language: str = "en"
        self.translator: Translator = Translator()
        self.alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self._translated_prompt_parts: dict = {}
        self._check_nltk_punkt()
        # self.tokenizer (transformers.AutoTokenizer) não é mais um atributo principal da classe para tokenização.
        # Será carregado apenas se necessário para o fallback de model_max_length.
        self.model_max_length: int = self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS

    def _check_nltk_punkt(self):
        """Checks if the NLTK 'punkt' tokenizer models are available."""
        try:
            sent_tokenize("Test sentence for NLTK punkt check.")
        except LookupError:
            hlog("WARNING: NLTK 'punkt' package not found. Needed for sentence tokenization.")
            hlog("WARNING: Please download it by running: import nltk; nltk.download('punkt')")

    def evaluate(
        self,
        executor, # helm.benchmark.executor.Executor
        benchmark_path: str,
        scenario_state, # helm.benchmark.runner.ScenarioState
        language: str,
        tokenizer_service: TokenizerService # MODIFICADO: Recebe TokenizerService
    ) -> list:
        """Public method to initiate contamination evaluation."""
        return asyncio.run(self._evaluate_async(
            executor,
            benchmark_path,
            scenario_state,
            language,
            tokenizer_service # MODIFICADO: Passa TokenizerService
        ))

    async def _get_or_translate_prompt_parts(self) -> tuple:
        """
        Retrieves static prompt parts, translating them once per language and caching the result.
        """
        if self.language in self._translated_prompt_parts:
            return self._translated_prompt_parts[self.language]

        base_parts = (
            "Please fill in the [] in option",
            "based on your benchmark knowledge.",
            "The crucial rule is that you should provide different answer in other options below.",
            "Question:",
            "Options:",
            "Reply with the answer only."
        )

        if self.language == "en" or not self.language:
            self._translated_prompt_parts[self.language] = base_parts
            return base_parts

        try:
            hlog(f"Translating prompt parts to: {self.language} for the first time for this evaluator instance.")
            loop = asyncio.get_event_loop()
            tasks = [loop.run_in_executor(None, self._blocking_translate_text, part) for part in base_parts]
            translation_results = await asyncio.gather(*tasks, return_exceptions=True)

            final_parts = tuple(
                res if not isinstance(res, Exception) else base_parts[i]
                for i, res in enumerate(translation_results)
            )
            self._translated_prompt_parts[self.language] = final_parts
            return final_parts
        except Exception as e:
            hlog(f"Error during translation process: {e}. Using original English text for all parts.")
            self._translated_prompt_parts[self.language] = base_parts
            return base_parts

    def _blocking_translate_text(self, text: str) -> str:
        """Synchronous translation call."""
        return self.translator.translate(text, src="en", dest=self.language).text

    async def _evaluate_async(
        self,
        executor, # helm.benchmark.executor.Executor
        benchmark_path: str,
        scenario_state, # helm.benchmark.runner.ScenarioState
        language: str,
        tokenizer_service: TokenizerService # MODIFICADO: Recebe e usará TokenizerService
    ) -> list:
        self.language = language
        with htrack_block("TS-Guessing (question-multichoice) contamination evaluation"):
            eval_data_name = os.path.basename(benchmark_path).split(":")[0]
            if not (
                hasattr(scenario_state, "adapter_spec")
                and hasattr(scenario_state.adapter_spec, "method")
                and scenario_state.adapter_spec.method == "multiple_choice_joint"
            ):
                hlog(f'The selected benchmark "{eval_data_name}" does not qualify for this contamination detection strategy.')
                return []

            model_id_from_spec = scenario_state.adapter_spec.model_deployment or scenario_state.adapter_spec.model
            if not model_id_from_spec:
                hlog("ERROR: Model identifier (model_deployment or model) not found in AdapterSpec. Cannot proceed.")
                return []

            # MODIFICADO: Obtenção do model_max_length prioritariamente do HELM Client
            try:
                # executor.adapter.client é o caminho usual para o Client no HELM
                client_for_model = executor.adapter.client
                self.model_max_length = client_for_model.get_max_context_length(model_id_from_spec)
                hlog(f"Using model_max_length from HELM client: {self.model_max_length} for {model_id_from_spec}")
            except Exception as client_exc:
                hlog(f"WARNING: Could not get max_context_length from HELM client for '{model_id_from_spec}': {client_exc}. "
                     "Falling back to transformers.AutoTokenizer or default.")
                # Fallback: Tentar carregar transformers.AutoTokenizer apenas para model_max_length
                temp_tokenizer_for_max_len: AutoTokenizer | None = None
                try:
                    # Usa o model_id_from_spec completo, que AutoTokenizer geralmente lida bem.
                    # A normalização para 'gpt2' pode ser adicionada se houver problemas específicos.
                    hlog(f"Fallback: Loading transformers.AutoTokenizer for: {model_id_from_spec} to get model_max_length.")
                    temp_tokenizer_for_max_len = AutoTokenizer.from_pretrained(model_id_from_spec, trust_remote_code=True)
                    tokenizer_max_len_attr = getattr(temp_tokenizer_for_max_len, 'model_max_length', self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS)

                    if not isinstance(tokenizer_max_len_attr, int) or tokenizer_max_len_attr > 200000:
                        hlog(f"WARNING: Fallback tokenizer for {model_id_from_spec} reported an unusual max length ({tokenizer_max_len_attr}). "
                             f"Using default: {self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS}.")
                        self.model_max_length = self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS
                    else:
                        self.model_max_length = tokenizer_max_len_attr
                except Exception as e_hf_tokenizer:
                    hlog(f"ERROR: Fallback using transformers.AutoTokenizer for '{model_id_from_spec}' also failed: {e_hf_tokenizer}. "
                         f"Using default model_max_length: {self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS}.")
                    self.model_max_length = self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS
                finally:
                    del temp_tokenizer_for_max_len # Liberar memória se carregado
                hlog(f"Using model_max_length (from fallback): {self.model_max_length} for {model_id_from_spec}")

            if self.model_max_length <= self.SMALL_MODEL_CONTEXT_THRESHOLD:
                hlog(f"WARNING: Model {model_id_from_spec} (effective context window {self.model_max_length} tokens) "
                     "may skip many instances if prompts are too long.")

            data_points = self._filter_data(scenario_state)
            hlog(f"Filtered to {len(data_points)} data points.")
            if not data_points: return []

            shuffled_data_points = [data_points[i] for i in np.random.permutation(len(data_points))]

            reference_texts_for_masked_slots = []
            masked_option_letters = []
            valid_request_states_for_execution = []
            skipped_instance_count = 0
            processed_instance_count = 0

            prompt_components = await self._get_or_translate_prompt_parts()

            generation_adapter_spec = replace(
                scenario_state.adapter_spec,
                method="generation",
                instructions="",
                input_prefix="", output_prefix="Answer: ", input_suffix="", output_suffix="",
                max_tokens=self.MAX_OUTPUT_TOKENS,
                temperature=self.GENERATION_TEMPERATURE,
                stop_sequences=[], num_outputs=1,
                global_prefix="", global_suffix="", reference_prefix="", reference_suffix="",
            )

            for data_point_item in shuffled_data_points:
                original_idx = data_point_item["original_request_state_index"]
                if original_idx >= len(scenario_state.request_states):
                    hlog(f"DEBUG: original_request_state_index {original_idx} out of bounds for initial scenario_state. Skipping.")
                    continue

                current_request_state = scenario_state.request_states[original_idx]
                try:
                    # MODIFICADO: _build_prompt não precisa mais de self.tokenizer
                    instruction_text, user_text, original_text_of_masked_option, wrong_letter = self._build_prompt(
                        data_point_item, prompt_components
                    )

                    if instruction_text == "failed":
                        continue

                    combined_prompt = f"{instruction_text}\n\n{user_text}"

                    # MODIFICADO: Uso do tokenizer_service para contar tokens
                    try:
                        tokenization_request = TokenizationRequest(text=combined_prompt, tokenizer=model_id_from_spec)
                        tokenization_result: TokenizationRequestResult = tokenizer_service.tokenize(tokenization_request)
                        num_prompt_tokens = len(tokenization_result.tokens)
                    except Exception as e_tok_service:
                        hlog(f"ERROR: TokenizerService failed to tokenize prompt for instance {original_idx}. Skipping. Error: {e_tok_service} - {traceback.format_exc()}")
                        skipped_instance_count += 1
                        continue


                    max_allowable_prompt_tokens = self.model_max_length - self.MAX_OUTPUT_TOKENS - self.TOKENIZER_BUFFER

                    if num_prompt_tokens > max_allowable_prompt_tokens:
                        skipped_instance_count += 1
                        # hlog(f"DEBUG: Skipping instance {original_idx} due to prompt length. Tokens: {num_prompt_tokens}, Max allowed: {max_allowable_prompt_tokens}")
                        continue

                    new_input = replace(current_request_state.instance.input, text=combined_prompt)
                    new_instance = replace(
                        current_request_state.instance,
                        input=new_input,
                        references=[]
                    )
                    new_request = replace(
                        current_request_state.request,
                        prompt=combined_prompt,
                        max_tokens=self.MAX_OUTPUT_TOKENS,
                        temperature=self.GENERATION_TEMPERATURE
                    )

                    prepared_rs = replace(
                        current_request_state,
                        instance=new_instance,
                        request=new_request,
                        result=None
                    )

                    if hasattr(prepared_rs, "output_mapping"):
                         prepared_rs = replace(prepared_rs, output_mapping=None)
                    if hasattr(prepared_rs, "reference_index"):
                         prepared_rs = replace(prepared_rs, reference_index=None)


                    valid_request_states_for_execution.append(prepared_rs)
                    reference_texts_for_masked_slots.append(original_text_of_masked_option)
                    masked_option_letters.append(wrong_letter)
                    processed_instance_count +=1

                except Exception as e:
                    hlog(f"Error preparing instance for original_request_state_index {original_idx}: {e} - {traceback.format_exc()}")

            if skipped_instance_count > 0:
                hlog(f"INFO: Skipped {skipped_instance_count} out of {len(shuffled_data_points)} instances due to prompt length exceeding model capacity or tokenization errors.")

            if not valid_request_states_for_execution:
                hlog("No instances were prepared for execution.")
                return []

            hlog(f"Sending {len(valid_request_states_for_execution)} prepared requests to the model.")

            execution_scenario_state = replace(
                scenario_state,
                adapter_spec=generation_adapter_spec,
                request_states=valid_request_states_for_execution
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
                        # MODIFICADO: _process_response não precisa mais de self.tokenizer (nunca precisou diretamente)
                        processed_response = self._process_response(response_text, masked_option_letters[i])
                        instance_id = getattr(valid_request_states_for_execution[i].instance, "id", f"processed_instance_{i}")
                        processed_instance_results.append({
                            "id": instance_id,
                            "answer_to_compare": reference_texts_for_masked_slots[i].lower(),
                            "model_prediction": processed_response.lower()
                        })
                except Exception as e:
                    hlog(f"Error processing result for prepared_instance_index {i}: {e}")

            if not processed_instance_results:
                hlog("No valid results to evaluate after model query.")
                return []

            gold_references = [res["answer_to_compare"] for res in processed_instance_results]
            model_generations = [res["model_prediction"] for res in processed_instance_results]

            calculated_metrics = {
                "exact_match": sum(gen == gold for gen, gold in zip(model_generations, gold_references)) / len(model_generations) if model_generations else 0.0,
                "rouge_l_f1": 0.0
            }

            if model_generations:
                try:
                    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
                    rouge_scores_f1 = []
                    for i_score in range(len(model_generations)):
                        try:
                            score = scorer.score(str(model_generations[i_score]), str(gold_references[i_score]))["rougeLsum"].fmeasure
                            rouge_scores_f1.append(score)
                        except Exception:
                            rouge_scores_f1.append(0.0)
                    if rouge_scores_f1: calculated_metrics["rouge_l_f1"] = np.mean(rouge_scores_f1)
                except Exception as e:
                    hlog(f"ROUGE Scorer initialization or general calculation error: {e}")

            final_helm_stats = []
            for metric_name, metric_value in calculated_metrics.items():
                metric_value = round(metric_value, 2)
                final_helm_stats.append({
                    "name": {"name": f"contamination (ts guessing multichoice {metric_name})", "split": "test"},
                    "count": 1,
                    "sum": metric_value,
                    "sum_squared": metric_value**2,
                    "min": metric_value,
                    "max": metric_value,
                    "mean": metric_value,
                    "variance": 0.0,
                    "stddev": 0.0
                })
            hlog(f"Evaluation completed. Processed {len(processed_instance_results)} instances. Skipped {skipped_instance_count} instances due to length or tokenization errors.")
            return final_helm_stats

    def _filter_data(self, scenario_state):
        """Extracts and validates data points from the scenario state."""
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

    # MODIFICADO: _build_prompt não depende mais de self.tokenizer para truncamento
    def _build_prompt(self, example_item, prompt_components_tuple):
        """
        Constructs the instruction and user text for a given example.
        The 'original_text_of_masked_option' returned is the original text of the
        randomly chosen *incorrect* option that was masked.
        No internal tokenization or truncation happens here; the full prompt is constructed.
        """
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

            instruction_text_segment = f"{part_instr_fill} {masked_option_letter} {part_instr_knowledge}\n\n{part_instr_rule}"

            options_lines_segment = ""
            for i, choice_text_item in enumerate(choices_list):
                if i >= len(self.alphabet): continue # Skip if more options than alphabet letters
                letter = self.alphabet[i]
                option_content = "[MASK]" if i == masked_choice_idx else f"[{str(choice_text_item)}]"
                options_lines_segment += f"\n{letter}: {option_content}"

            # O prompt é construído com o texto original completo da pergunta.
            # A checagem de comprimento e o possível "pulo" da instância ocorrerão em _evaluate_async.
            user_text_segment = f"{part_header_q} {original_question_text}\n{part_header_opts}{options_lines_segment}\n\n{part_footer_reply}"

            return instruction_text_segment, user_text_segment, original_text_of_masked_option, masked_option_letter
        except Exception as e:
            hlog(f"EXCEPTION: Critical error in _build_prompt for example {example_item.get('id', 'Unknown ID')}: {e} - {traceback.format_exc()}")
            return "failed", "", "", ""

    def _process_response(self, response_text_input, masked_option_letter_param):
        """Cleans and extracts the relevant part of the model's response."""
        try:
            if not response_text_input: return ""

            processed_text = str(response_text_input)
            option_prefix_to_strip = masked_option_letter_param + ":"

            if option_prefix_to_strip in processed_text:
                parts = processed_text.split(option_prefix_to_strip, 1)
                processed_text = parts[1].strip() if len(parts) > 1 else processed_text

            try:
                sentences = sent_tokenize(processed_text)
                if sentences: processed_text = sentences[0]
            except LookupError:
                # Warning already issued by _check_nltk_punkt if not found initially
                pass
            except Exception:
                pass

            processed_text = processed_text.replace("[MASK]", "").strip()
            if processed_text.startswith("[") and processed_text.endswith("]"):
                processed_text = processed_text[1:-1].strip()

            return processed_text
        except Exception as e:
            hlog(f"Error processing response: {e}")
            return ""

    async def _query_model(self, scenario_state, executor):
        """Executes the requests against the model via the HELM executor."""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, executor.execute, scenario_state)
        except Exception as e:
            hlog(f"Model query failed: {e}")
            raise e