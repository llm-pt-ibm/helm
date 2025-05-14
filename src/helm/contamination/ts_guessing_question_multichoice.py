import os
import numpy as np
import asyncio
import traceback
from googletrans import Translator
from rouge_score import rouge_scorer
from dataclasses import replace
import nltk

from helm.common.hierarchical_logger import hlog, htrack_block
from nltk.tokenize import sent_tokenize
from .utils_contamination import UtilsContamination 
from transformers import AutoTokenizer

class TSGuessingQuestionMultiChoiceContaminationEvaluator:
    """
    Implements a question-based multi-choice guessing test for contamination detection.
    """

    DEFAULT_MODEL_MAX_CONTEXT_TOKENS = 1024
    MAX_OUTPUT_TOKENS = 100        
    TOKENIZER_BUFFER = 30          

    def __init__(self):
        self.language: str = "en"
        self.translator: Translator = Translator()
        self.alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self._translated_prompt_parts: dict = {}
        self._check_nltk_punkt()
        self.tokenizer: AutoTokenizer | None = None
        self.model_max_length: int = self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS

    def evaluate(
        self,
        executor,
        benchmark_path: str,
        scenario_state,
        language: str
    ) -> list:
        """Public method to initiate contamination evaluation."""
        return asyncio.run(self._evaluate_async(
            executor,
            benchmark_path,
            scenario_state,
            language
        ))

    def _check_nltk_punkt(self):
        """Checks if the NLTK 'punkt' tokenizer models are available."""
        try:
            sent_tokenize("Test sentence for NLTK punkt check.")
        except LookupError:
            hlog("WARNING: NLTK 'punkt' package not found. Needed for sentence tokenization.")
            hlog("WARNING: Please download it by running: import nltk; nltk.download('punkt')")

    async def _get_or_translate_prompt_parts(self) -> tuple:
        """
        Retrieves static prompt parts, translating them once per language and caching the result.
        """
        if self.language in self._translated_prompt_parts:
            return self._translated_prompt_parts[self.language]

        # Standard English prompt components
        base_parts = (
            "Please fill in the [] in option",
            "based on your benchmark knowledge.",
            "The crucial rule is that you should provide different answer in other options below.", 
            "Question:",
            "Options:",
            "Reply with the answer only."
        )

        if self.language == "en" or not self.language: # Default to English if no language or 'en'
            self._translated_prompt_parts[self.language] = base_parts
            return base_parts

        try:
            hlog(f"Translating prompt parts to: {self.language} for the first time for this evaluator instance.")
            loop = asyncio.get_event_loop()
            tasks = [loop.run_in_executor(None, self._blocking_translate_text, part) for part in base_parts]
            translation_results = await asyncio.gather(*tasks, return_exceptions=True)

            final_parts = [
                res if not isinstance(res, Exception) else base_parts[i]
                for i, res in enumerate(translation_results)
            ]
            self._translated_prompt_parts[self.language] = tuple(final_parts)
            return tuple(final_parts)
        except Exception as e:
            hlog(f"Error during translation process: {e}. Using original English text for all parts.")
            self._translated_prompt_parts[self.language] = base_parts
            return base_parts

    def _blocking_translate_text(self, text: str) -> str:
        """Synchronous translation call."""
        return self.translator.translate(text, src="en", dest=self.language).text

    async def _evaluate_async(
        self,
        executor,
        benchmark_path: str,
        scenario_state,
        language: str
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

            tokenizer_id = model_id_from_spec
            if model_id_from_spec.lower() in ["huggingface/gpt2", "openai/gpt2"]: 
                tokenizer_id = "gpt2"

            try:
                if self.tokenizer is None or self.tokenizer.name_or_path != tokenizer_id:
                    hlog(f"Loading tokenizer for: {tokenizer_id} (derived from: {model_id_from_spec})")
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

                    if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                        hlog(f"DEBUG: Tokenizer for {tokenizer_id} has no pad_token_id. Setting pad_token_id to eos_token_id ({self.tokenizer.eos_token_id}).")
                        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

                    tokenizer_max_len = getattr(self.tokenizer, 'model_max_length', self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS)
                    if not isinstance(tokenizer_max_len, int) or tokenizer_max_len > 200000:
                        hlog(f"WARNING: Tokenizer for {tokenizer_id} reported max length {tokenizer_max_len}. Using default {self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS}.")
                        self.model_max_length = self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS
                    else:
                        self.model_max_length = tokenizer_max_len
                    hlog(f"Using model_max_length: {self.model_max_length} for {tokenizer_id}")
            except Exception as e:
                hlog(f"ERROR: Could not load tokenizer for '{tokenizer_id}' (derived from '{model_id_from_spec}'): {e}. Cannot perform token-based truncation.")
                return []

            data_points = self._filter_data(scenario_state)
            hlog(f"Filtered to {len(data_points)} data points.")
            if not data_points: return []

            shuffled_data_points = [data_points[i] for i in np.random.permutation(len(data_points))]

            reference_masked_option_texts = []
            masked_option_letters = []
            prepared_request_original_indices = []

            prompt_components = await self._get_or_translate_prompt_parts()
            original_adapter_spec = scenario_state.adapter_spec

            try:
                scenario_state.adapter_spec = replace(
                    original_adapter_spec, method="generation", instructions="",
                    input_prefix="", input_suffix="", output_prefix="Answer: ",
                    max_tokens=self.MAX_OUTPUT_TOKENS
                )
                for data_point_item in shuffled_data_points:
                    original_idx = data_point_item["original_request_state_index"]
                    if original_idx >= len(scenario_state.request_states): continue

                    current_request_state = scenario_state.request_states[original_idx]
                    try:
                        instruction_text, user_text, original_masked_text, wrong_letter = self._build_prompt(data_point_item, prompt_components)
                        if instruction_text != "failed":
                            combined_prompt = f"{instruction_text}\n\n{user_text}"
                            new_input = replace(current_request_state.instance.input, text=combined_prompt)
                            new_instance = replace(current_request_state.instance, input=new_input, references=[])
                            new_request = replace(current_request_state.request, prompt=combined_prompt, max_tokens=self.MAX_OUTPUT_TOKENS)

                            scenario_state.request_states[original_idx] = replace(
                                current_request_state, instance=new_instance,
                                request=new_request, result=None
                            )
                            reference_masked_option_texts.append(original_masked_text)
                            masked_option_letters.append(wrong_letter)
                            prepared_request_original_indices.append(original_idx)
                    except Exception as e:
                        hlog(f"Error building prompt for original_request_state_index {original_idx}: {e}")

                if not prepared_request_original_indices:
                    hlog("No valid prompts were constructed.")
                    return []

                response_scenario_state = await self._query_model(scenario_state, executor)
            except Exception as e:
                hlog(f"Error during prompt preparation or model query: {e}")
                return []
            finally:
                scenario_state.adapter_spec = original_adapter_spec

            processed_instance_results = []
            for i, original_idx in enumerate(prepared_request_original_indices):
                if original_idx >= len(response_scenario_state.request_states): continue

                response_state = response_scenario_state.request_states[original_idx]
                try:
                    if response_state.result and response_state.result.completions:
                        response_text = response_state.result.completions[0].text.strip()
                        processed_response = self._process_response(response_text, masked_option_letters[i])
                        processed_instance_results.append({
                            "id": f"instance_{original_idx}",
                            "answer": reference_masked_option_texts[i].lower(),
                            "response": processed_response.lower()
                        })
                except Exception as e:
                    hlog(f"Error processing result for original_request_state_index {original_idx}: {e}")

            if not processed_instance_results:
                hlog("No valid results to evaluate after model query.")
                return []

            gold_answers = [res["answer"] for res in processed_instance_results]
            model_responses = [res["response"] for res in processed_instance_results]

            calculated_metrics = {
                "exact_match": sum(res == ans for res, ans in zip(model_responses, gold_answers)) / len(model_responses) if model_responses else 0.0,
                "rouge_l": 0.0
            }

            if model_responses:
                try:
                    scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
                    rouge_scores = []
                    for i in range(len(model_responses)):
                        try:
                            score = scorer.score(str(model_responses[i]), str(gold_answers[i]))["rougeLsum"].fmeasure
                            rouge_scores.append(score)
                        except Exception as rouge_exc:
                            hlog(f"DEBUG: ROUGE score calculation error for instance {processed_instance_results[i]['id']}: Response='{model_responses[i]}', Gold='{gold_answers[i]}'. Error: {rouge_exc}")
                            rouge_scores.append(0.0)
                    if rouge_scores: calculated_metrics["rouge_l"] = np.mean(rouge_scores)
                except Exception as e:
                    hlog(f"ROUGE calculation error: {e}")

            final_helm_stats = []
            for metric_name, metric_value in calculated_metrics.items():
                final_helm_stats.append({
                    "name": {"name": f"contamination (ts guessing multichoice {metric_name})", "split": "test"},
                    "count": 1,
                    "sum": metric_value,
                    "sum_squared": (metric_value**2),
                    "min": metric_value,
                    "max": metric_value,
                    "mean": metric_value,
                    "variance": 0.0,
                    "stddev": 0.0
                })
            return final_helm_stats

    def _filter_data(self, scenario_state):
        """Extracts and validates data points from the scenario state."""
        data_points = []
        for i, request_state_item in enumerate(scenario_state.request_states):
            try:
                question_text = UtilsContamination.get_question_text(request_state_item.instance)
                choices_list = UtilsContamination.get_choices(request_state_item.instance)
                correct_answer_idx = UtilsContamination.get_answer_index(request_state_item.instance)

                if question_text and choices_list and isinstance(correct_answer_idx, int) and 0 <= correct_answer_idx < len(choices_list):
                    data_points.append({
                        "id": getattr(request_state_item.instance, "id", f"instance_{i}"),
                        "text": question_text,
                        "choices": choices_list,
                        "answer_index": correct_answer_idx,
                        "original_request_state_index": i
                    })
            except Exception as e:
                hlog(f"Error filtering data point at index {i}: {e}")
        return data_points

    def _build_prompt(self, example_item, prompt_components_tuple):
        """Constructs the instruction and user text for a given example, with token-based truncation."""
        if not self.tokenizer:
            hlog("ERROR: Tokenizer is not available in _build_prompt. Cannot perform token-based truncation.")
            return "failed", "", "", ""
        try:
            (part_instr_fill, part_instr_knowledge, part_instr_rule,
             part_header_q, part_header_opts, part_footer_reply) = prompt_components_tuple

            original_question_text = example_item.get("text", "")
            choices_list = example_item.get("choices", [])
            correct_answer_idx = example_item.get("answer_index", -1)

            if not (original_question_text and choices_list and isinstance(correct_answer_idx, int) and 0 <= correct_answer_idx < len(choices_list)):
                return "failed", "", "", ""

            incorrect_option_indices = [i for i in range(len(choices_list)) if i != correct_answer_idx]
            if not incorrect_option_indices: return "failed", "", "", ""

            masked_choice_idx = np.random.choice(incorrect_option_indices)
            original_masked_option_text = str(choices_list[masked_choice_idx])
            masked_option_letter = self.alphabet[masked_choice_idx % len(self.alphabet)]

            instruction_text_segment = f"{part_instr_fill} {masked_option_letter} {part_instr_knowledge}\n\n{part_instr_rule}"

            options_lines_segment = ""
            for i, choice_text_item in enumerate(choices_list):
                if i >= len(self.alphabet): continue
                letter = self.alphabet[i]
                option_content = "[MASK]" if i == masked_choice_idx else f"[{str(choice_text_item)}]"
                options_lines_segment += f"\n{letter}: {option_content}"

            fixed_parts_tokens = sum(
                len(self.tokenizer.encode(text_part, add_special_tokens=False))
                for text_part in [
                    instruction_text_segment, f"\n\n{part_header_q} ",
                    f"\n{part_header_opts}", options_lines_segment, f"\n\n{part_footer_reply}"
                ]
            )

            question_tokens_budget = self.model_max_length - fixed_parts_tokens - self.MAX_OUTPUT_TOKENS - self.TOKENIZER_BUFFER
            question_tokens_budget = max(10, question_tokens_budget)

            original_question_tokens = self.tokenizer.encode(original_question_text, add_special_tokens=False)
            final_question_text = original_question_text

            if len(original_question_tokens) > question_tokens_budget:
                tokens_to_remove_count = len(original_question_tokens) - question_tokens_budget
                remove_from_first_half_end = tokens_to_remove_count // 2
                remove_from_second_half_start = tokens_to_remove_count - remove_from_first_half_end
                mid_point_token_idx = len(original_question_tokens) // 2
                first_part_end_idx = max(0, mid_point_token_idx - remove_from_first_half_end)
                second_part_start_idx = min(len(original_question_tokens), mid_point_token_idx + remove_from_second_half_start)

                if first_part_end_idx < second_part_start_idx :
                    first_half_tokens_truncated = original_question_tokens[:first_part_end_idx]
                    second_half_tokens_truncated = original_question_tokens[second_part_start_idx:]
                    part1_decoded_text = self.tokenizer.decode(first_half_tokens_truncated, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                    part2_decoded_text = self.tokenizer.decode(second_half_tokens_truncated, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                    if part1_decoded_text and part2_decoded_text:
                        final_question_text = f"{part1_decoded_text} ... {part2_decoded_text}"
                    elif part1_decoded_text: final_question_text = part1_decoded_text + "..."
                    elif part2_decoded_text: final_question_text = "... " + part2_decoded_text
                    else: final_question_text = self.tokenizer.decode(original_question_tokens[:question_tokens_budget], skip_special_tokens=True, clean_up_tokenization_spaces=True).strip() + "..."
                else:
                    hlog(f"DEBUG: Middle truncation points invalid for instance {example_item.get('id', 'Unknown')}. Truncating from end.")
                    safe_truncated_tokens = original_question_tokens[:question_tokens_budget]
                    final_question_text = self.tokenizer.decode(safe_truncated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True).strip()
                    if original_question_text and final_question_text != original_question_text: final_question_text += "..."

            user_text_segment = f"{part_header_q} {final_question_text}\n{part_header_opts}{options_lines_segment}\n\n{part_footer_reply}"
            return instruction_text_segment, user_text_segment, original_masked_option_text, masked_option_letter
        except Exception as e:
            hlog(f"EXCEPTION: Critical error in _build_prompt for example {example_item.get('id', 'Unknown ID')}: {e} - {traceback.format_exc()}")
            return "failed", "", "", ""

    def _process_response(self, response_text_input, wrong_option_letter):
        """Cleans and extracts the relevant part of the model's response."""
        try:
            if not response_text_input: return ""

            processed_text = str(response_text_input)
            option_prefix_to_strip = wrong_option_letter + ":"

            if option_prefix_to_strip in processed_text:
                parts = processed_text.split(option_prefix_to_strip, 1)
                processed_text = parts[1].strip() if len(parts) > 1 else processed_text

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