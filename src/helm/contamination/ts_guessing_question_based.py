import os
import numpy as np
import spacy
import asyncio
import traceback
from dataclasses import replace
from typing import List, Dict, Any, Tuple, Optional

from helm.common.hierarchical_logger import hlog, htrack_block
from helm.benchmark.scenarios.scenario import Instance 
from helm.common.tokenization_request import TokenizationRequest
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from .utils_contamination import UtilsContamination

class TSGuessingQuestionBasedContaminationEvaluator:
    """
    Implementation of the 'base' TS-Guessing strategy.
    Masks an important word (NOUN, ADJ, VERB) in a sentence and asks the model
    to predict the masked word. Contamination is measured via exact match.
    Reference: https://aclanthology.org/2024.naacl-long.482/
    """
    STRATEGY_NAME: str = "ts_guessing_base"
    STRATEGY_DISPLAY_NAME: str = "TS-Guessing (base/word-masking)"

    DEFAULT_MODEL_MAX_CONTEXT_TOKENS: int = 4096 
    SMALL_MODEL_CONTEXT_THRESHOLD: int = 1024
    MAX_OUTPUT_TOKENS: int = 20
    TOKENIZER_BUFFER: int = 30
    GENERATION_TEMPERATURE: float = 0.1
    POS_TAGS_TO_MASK: List[str] = ["NOUN", "ADJ", "VERB"]

    def __init__(self):
        self.language: str = "en"

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
        tagger: Optional[spacy.language.Language] = None

        with htrack_block(f"{self.STRATEGY_DISPLAY_NAME} contamination evaluation for language '{self.language}'"):
            try:
                tagger = UtilsContamination.get_spacy_tagger(self.language)
            except Exception as e:
                hlog(f"STRATEGY CRITICAL: Failed to load spaCy tagger for language '{self.language}': {e}\n{traceback.format_exc()}")
                return []

            model_deployment_name_from_spec = scenario_state.adapter_spec.model_deployment or scenario_state.adapter_spec.model
            if not model_deployment_name_from_spec:
                hlog("STRATEGY ERROR: Model identifier (model_deployment or model) not found in AdapterSpec. Cannot proceed.")
                return []

            model_max_length = UtilsContamination.determine_model_max_length(
                model_deployment_name_from_spec,
                self.DEFAULT_MODEL_MAX_CONTEXT_TOKENS
            )
            hlog(f"STRATEGY INFO: Effective model_max_length for {model_deployment_name_from_spec}: {model_max_length}")

            if model_max_length <= self.SMALL_MODEL_CONTEXT_THRESHOLD:
                hlog(f"STRATEGY WARNING: Model {model_deployment_name_from_spec} (context window {model_max_length} tokens) "
                     "may skip many instances if prompts are too long.")

            raw_instances_from_state = [rs.instance for rs in scenario_state.request_states if hasattr(rs, "instance")]
            if not raw_instances_from_state:
                hlog("STRATEGY INFO: No raw instances found in scenario_state.request_states.")
                return []

            data_points = self._filter_data(raw_instances_from_state, scenario_state)
            hlog(f"STRATEGY INFO: Filtered to {len(data_points)} data points for evaluation.")
            if not data_points:
                hlog("STRATEGY INFO: No data points available after filtering.")
                return []

            shuffled_data_points = [data_points[i] for i in np.random.permutation(len(data_points))]

            original_masked_words: List[str] = []
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
                "output_prefix": prompt_components.get("answer_prefix", "Answer: "),
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
            
            max_allowable_prompt_tokens = model_max_length - self.MAX_OUTPUT_TOKENS_BASE - self.TOKENIZER_BUFFER_BASE

            for data_point_item in shuffled_data_points:
                original_rs_idx = data_point_item.get("original_request_state_index", -1)
                
                if not (0 <= original_rs_idx < len(scenario_state.request_states)):
                    hlog(f"STRATEGY DEBUG: original_request_state_index {original_rs_idx} out of bounds or missing. Skipping.")
                    skipped_instance_count += 1
                    continue
                
                current_request_state = scenario_state.request_states[original_rs_idx]

                try:
                    final_prompt_text, masked_word_original = self._build_prompt(
                        data_point_item, tagger, prompt_components
                    )
                    if final_prompt_text == "failed":
                        skipped_instance_count += 1
                        continue
                    
                    is_valid_len, num_prompt_tokens = UtilsContamination.check_prompt_length(
                        final_prompt_text, model_deployment_name_from_spec, tokenizer_service, max_allowable_prompt_tokens
                    )
                    if not is_valid_len:
                        hlog(f"STRATEGY DEBUG: Instance from original_rs_idx {original_rs_idx} skipped. Prompt too long ({num_prompt_tokens} > {max_allowable_prompt_tokens}).")
                        skipped_instance_count += 1
                        continue
                        
                    new_input = replace(current_request_state.instance.input, text=final_prompt_text)
                    new_instance = replace(current_request_state.instance, input=new_input, references=[])
                    
                    new_request = replace(
                        current_request_state.request,
                        prompt=final_prompt_text,
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
                    original_masked_words.append(masked_word_original.lower())

                except Exception as e:
                    hlog(f"STRATEGY ERROR: Error preparing instance from original_rs_idx {original_rs_idx}: {e}\n{traceback.format_exc()}")
                    skipped_instance_count += 1
            
            if skipped_instance_count > 0:
                hlog(f"STRATEGY INFO: Skipped {skipped_instance_count} instances during preparation.")
            if not valid_request_states_for_execution:
                hlog("STRATEGY INFO: No instances prepared for execution.")
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
                        full_response_text = response_state.result.completions[0].text.strip()
                        processed_model_word = self._process_response(full_response_text)
                        
                        instance_id = getattr(valid_request_states_for_execution[i].instance, "id", f"base_inst_{i}")
                        processed_instance_results.append({
                            "id": instance_id,
                            "masked_word_original": original_masked_words[i],
                            "model_predicted_word": processed_model_word.lower() 
                        })
                    else:
                         hlog(f"STRATEGY WARNING: No valid completion for prepared instance index {i}.")
                except Exception as e:
                    hlog(f"STRATEGY ERROR: Error processing result for prepared instance index {i}: {e}\n{traceback.format_exc()}")

            if not processed_instance_results:
                hlog("STRATEGY INFO: No valid results after model query and processing.")
                return []

            exact_match_count = sum(
                1 for res in processed_instance_results if res["model_predicted_word"] == res["masked_word_original"]
            )
            exact_match_score = (exact_match_count / len(processed_instance_results)) if processed_instance_results else 0.0
            
            calculated_metrics = {"exact_match": exact_match_score}
            
            strategy_metric_prefix = f"contamination ({self.STRATEGY_NAME}"
            final_helm_stats = UtilsContamination.format_helm_stats(
                calculated_metrics, strategy_metric_prefix, split="test"
            )

            hlog(f"STRATEGY INFO: Evaluation completed for language '{self.language}'. "
                 f"Processed {len(processed_instance_results)} instances. "
                 f"Final Exact Match: {exact_match_score:.2f}")
            return final_helm_stats

    def _filter_data(self, raw_instances: List[Instance], scenario_state_for_context) -> List[Dict[str, Any]]:
        data_points: List[Dict[str, Any]] = []

        for i, instance in enumerate(raw_instances):
            try:
                text_content = UtilsContamination.get_question_text(instance)

                if text_content and text_content != "Unknown question or context":
                    words_in_text = text_content.split()
                    if len(words_in_text) > 4:
                        data_points.append({
                            "text_to_mask": text_content,
                            "original_request_state_index": scenario_state_for_context.request_states.index(
                                next(rs for rs in scenario_state_for_context.request_states if rs.instance == instance)
                            )
                        })
                    else:
                        hlog(f"STRATEGY DEBUG: Skipping instance {i} due to insufficient word count in text: '{text_content[:50]}...'")
                else:
                    hlog(f"STRATEGY DEBUG: Skipping instance {i} due to missing or invalid text_content.")
            except StopIteration:
                hlog(f"STRATEGY WARNING: Could not find original request_state for instance {i}. Skipping.")
            except Exception as e:
                hlog(f"STRATEGY ERROR: Error filtering instance {i}: {e}\n{traceback.format_exc()}")
        return data_points

    def _build_prompt(
        self,
        example_item: Dict[str, Any],
        tagger: spacy.language.Language,
        prompt_components: Dict[str, str]
    ) -> Tuple[str, str]:
        try:
            text_to_process = example_item.get("text_to_mask")
            if not text_to_process or not isinstance(text_to_process, str):
                hlog("STRATEGY DEBUG: _build_prompt failed - invalid text_to_process.")
                return "failed", ""

            doc = tagger(text_to_process)
            candidate_words = [token for token in doc if token.pos_ in self.POS_TAGS_TO_MASK and not token.is_stop and not token.is_punct and len(token.text) > 2]

            if not candidate_words:
                hlog(f"STRATEGY DEBUG: No suitable words (NOUN, ADJ, VERB, non-stop, non-punct, len > 2) found in text: '{text_to_process[:100]}...'")
                return "failed", ""

            selected_token = np.random.choice(candidate_words)
            word_to_mask_original_case = selected_token.text
            
            try:
                import re
                escaped_word = re.escape(word_to_mask_original_case)
                masked_text = re.sub(r'\b' + escaped_word + r'\b', "[MASK]", text_to_process, 1, flags=re.IGNORECASE if self.language != "zh" else 0)
                
                if masked_text == text_to_process:
                    hlog(f"STRATEGY WARNING: Word '{word_to_mask_original_case}' not found for masking via regex in text (idx {selected_token.idx}). Falling back to simple replace. Text: '{text_to_process[:100]}...'")
                    start_char = selected_token.idx
                    end_char = start_char + len(word_to_mask_original_case)
                    masked_text = text_to_process[:start_char] + "[MASK]" + text_to_process[end_char:]

            except Exception as e_replace:
                 hlog(f"STRATEGY ERROR: Error replacing word '{word_to_mask_original_case}' in text: {e_replace}. Text: '{text_to_process[:100]}...'")
                 return "failed", ""


            instruction = prompt_components.get("instruction", "Fill in the [MASK]:")
            sentence_formatted = prompt_components.get("sentence_template", "\"{masked_sentence}\"").format(masked_sentence=masked_text)
            
            final_prompt = f"{instruction}\n{sentence_formatted}"
            
            return final_prompt, word_to_mask_original_case
        
        except Exception as e:
            hlog(f"STRATEGY EXCEPTION: Critical error in _build_prompt for item: {example_item.get('id', 'Unknown')}: {e}\n{traceback.format_exc()}")
            return "failed", ""

    def _process_response(self, full_response_text: str) -> str:
        try:
            if not full_response_text:
                return ""
            
            processed_text = str(full_response_text).strip()

            prefixes_to_strip = ["answer is", "answer:", "the word is", "it is", "it's"]
            for prefix in prefixes_to_strip:
                if processed_text.lower().startswith(prefix):
                    processed_text = processed_text[len(prefix):].strip()

            if self.language == "zh":
                if len(processed_text) <= 10:
                    first_word_or_phrase = processed_text
                else:
                    first_word_or_phrase = processed_text.split()[0] if processed_text.split() else processed_text[:10]
            else:
                response_words = processed_text.split()
                first_word_or_phrase = response_words[0] if response_words else ""
            
            punctuation_to_strip = '"""\'\'.,;!?¿¡#()[]{}<>'
            if self.language == "zh":
                punctuation_to_strip += '。，！？；：（）《》「」『』“”‘’'
            
            cleaned_word = first_word_or_phrase.strip(punctuation_to_strip)
            
            return cleaned_word
        except Exception as e:
            hlog(f"STRATEGY ERROR: Error processing model response (base strategy): '{full_response_text[:50]}...': {e}\n{traceback.format_exc()}")
            return ""

    async def _query_model(self, scenario_state, executor) -> Any:
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, executor.execute, scenario_state)
        except Exception as e:
            hlog(f"STRATEGY CRITICAL: Model query execution failed: {e}\n{traceback.format_exc()}")
            raise