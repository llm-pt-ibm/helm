import spacy
import numpy as np
from dataclasses import replace
from typing import Dict, List, Tuple, Any

from helm.common.hierarchical_logger import hlog
from helm.benchmark.model_deployment_registry import get_model_deployment, ModelDeployment
from helm.common.tokenization_request import TokenizationRequest, TokenizationRequestResult
from helm.benchmark.window_services.tokenizer_service import TokenizerService
from .prompt_translations import TS_GUESSING_BASE, TS_GUESSING_MULTICHOICE


PROMPT_CONFIGS_MASTER = {
    "ts_guessing_base": TS_GUESSING_BASE,
    "ts_guessing_multichoice": TS_GUESSING_MULTICHOICE
}

class UtilsContamination:
    """
    Utilities for contamination detection strategies.
    Provides common functionalities for handling HELM objects, model configurations,
    prompt management, and result formatting.
    """

    DEFAULT_MODEL_MAX_CONTEXT_TOKENS_UTIL: int = 4096
    SPACY_MODEL_MAP: Dict[str, str] = {
        "en": "en_core_web_sm",
        "pt": "pt_core_news_sm",
        "zh": "zh_core_web_sm",
    }

    @staticmethod
    def get_choices(example: Any) -> List[str]:
        """
        Extracts a list of choices from a HELM Instance or a dictionary.
        Tries various common structures for multiple-choice questions.
        """
        if hasattr(example, "references") and example.references:
            return [ref.output.text for ref in example.references if hasattr(ref, "output") and hasattr(ref.output, "text")]

        if hasattr(example, "output_mapping") and example.output_mapping and isinstance(example.output_mapping, dict):
            return list(example.output_mapping.values())

        if isinstance(example, dict):
            if "choices" in example:
                if isinstance(example["choices"], dict) and "text" in example["choices"] and isinstance(example["choices"]["text"], list):
                    return example["choices"]["text"]
                elif isinstance(example["choices"], list):
                    return example["choices"]
            if "endings" in example and isinstance(example["endings"], list):
                return example["endings"]
            if "options" in example and isinstance(example["options"], list):
                return example["options"]

        for i in range(1, 5):
            opt_key = f"option{i}"
            if hasattr(example, opt_key):
                opts = [getattr(example, f"option{j}") for j in range(1,5) if hasattr(example, f"option{j}")]
                return opts
            elif isinstance(example, dict) and opt_key in example:
                opts = [example[f"option{j}"] for j in range(1,5) if f"option{j}" in example]
                return opts
        
        hlog(f"UTIL WARNING: Could not extract choices from example: {type(example)}")
        return []

    @staticmethod
    def get_answer_index(example: Any) -> int:
        """
        Extracts the 0-based index of the correct answer from a HELM Instance or dict.
        Handles various ways correct answers are specified.
        """
        alphabet = "abcdefghijklmnopqrstuvwxyz123456789"

        if hasattr(example, "references") and example.references:
            for i, ref in enumerate(example.references):
                if hasattr(ref, "tags") and "correct" in ref.tags:
                    return i

        if hasattr(example, "output_mapping") and example.output_mapping and \
           hasattr(example, "references") and example.references:
            correct_text_from_ref = None
            for ref in example.references:
                if hasattr(ref, "tags") and "correct" in ref.tags and \
                   hasattr(ref, "output") and hasattr(ref.output, "text"):
                    correct_text_from_ref = ref.output.text
                    break
            if correct_text_from_ref and isinstance(example.output_mapping, dict):
                for letter_or_idx, text_val in example.output_mapping.items():
                    if text_val == correct_text_from_ref:
                        try:
                            return int(letter_or_idx)
                        except ValueError:
                            if isinstance(letter_or_idx, str):
                                key_lower = letter_or_idx.lower()
                                if key_lower in alphabet:
                                    return alphabet.index(key_lower)
                        hlog(f"UTIL WARNING: Found correct text in output_mapping but key '{letter_or_idx}' is not a recognized index format.")


        if isinstance(example, dict):
            if "answerKey" in example:
                key = str(example["answerKey"]).lower()
                if key.isdigit():
                    return int(key) -1 if int(key) > 0 else 0
                elif len(key) == 1 and key in alphabet:
                    return alphabet.index(key)
            if "label" in example:
                try:
                    return int(example["label"])
                except ValueError:
                    hlog(f"UTIL WARNING: Could not convert 'label' field ('{example['label']}') to int.")
            if "answer" in example:
                if isinstance(example["answer"], int):
                    return example["answer"]
                if isinstance(example["answer"], str) and example["answer"].isdigit():
                    return int(example["answer"])

        hlog(f"UTIL WARNING: Could not determine answer index for example: {type(example)}")
        return -1

    @staticmethod
    def get_question_text(example: Any) -> str:
        """
        Extracts the main question text from a HELM Instance or dictionary.
        """
        if hasattr(example, "input") and hasattr(example.input, "text"):
            return example.input.text

        if isinstance(example, dict):
            for key in ["question", "text", "query", "prompt", "goal", "passage", "context", "sentence"]:
                if key in example and isinstance(example[key], str):
                    if isinstance(example[key], dict) and "stem" in example[key]: return example[key]["stem"]
                    return example[key]
            if 'input' in example and isinstance(example['input'], str): return example['input']

        hlog(f"UTIL WARNING: Could not extract question text from example: {type(example)}")
        return "Unknown question or context"

    @staticmethod
    def get_prompt_fragments(strategy_key: str, language: str) -> Dict[str, str]:
        """
        Loads prompt fragments for a given strategy and language.
        Falls back to English if the specified language is not found.
        """
        if strategy_key not in PROMPT_CONFIGS_MASTER:
            hlog(f"UTIL ERROR: Prompt configuration for strategy '{strategy_key}' not found.")
            return {}

        lang_prompts_for_strategy = PROMPT_CONFIGS_MASTER[strategy_key]
        normalized_lang = language.lower().split('_')[0]

        if normalized_lang in lang_prompts_for_strategy:
            return lang_prompts_for_strategy[normalized_lang]
        elif "en" in lang_prompts_for_strategy:
            hlog(f"UTIL WARNING: Language '{language}' (normalized to '{normalized_lang}') not found for strategy '{strategy_key}'. Falling back to English.")
            return lang_prompts_for_strategy["en"]
        else:
            hlog(f"UTIL ERROR: English fallback not found for strategy '{strategy_key}' when language '{language}' is missing.")
            return {}

    @staticmethod
    def determine_model_max_length(
        model_deployment_name: str,
        default_max_len: int = DEFAULT_MODEL_MAX_CONTEXT_TOKENS_UTIL
    ) -> int:
        """
        Determines the maximum sequence length for a model.
        Prioritizes ModelDeployment, then AutoTokenizer, then a default.
        """
        model_max_len = default_max_len
        primary_source_found = False
        try:
            model_deployment: ModelDeployment = get_model_deployment(model_deployment_name)
            if model_deployment.max_sequence_length is not None and model_deployment.max_sequence_length > 0:
                model_max_len = model_deployment.max_sequence_length
                primary_source_found = True
                hlog(f"Util: Using model_max_length from ModelDeployment.max_sequence_length: {model_max_len} for {model_deployment_name}")
            elif model_deployment.max_request_length is not None and model_deployment.max_request_length > 0:
                model_max_len = model_deployment.max_request_length
                primary_source_found = True
                hlog(f"Util: Using model_max_length from ModelDeployment.max_request_length: {model_max_len} for {model_deployment_name}")
            
            if not primary_source_found:
                hlog(f"UTIL WARNING: max_sequence_length/max_request_length not set or invalid in ModelDeployment for '{model_deployment_name}'.")
                raise ValueError("Relevant max length fields not in ModelDeployment or invalid.")

        except (ValueError, KeyError) as e_model_reg:
            hlog(f"UTIL INFO: Could not get max length from ModelDeployment for '{model_deployment_name}': {e_model_reg}. "
                 "Falling back to AutoTokenizer or default method.")
            temp_tokenizer_for_max_len = None
            try:
                from transformers import AutoTokenizer
                hlog(f"UTIL Fallback: Loading AutoTokenizer for: {model_deployment_name} to get model_max_length.")
                temp_tokenizer_for_max_len = AutoTokenizer.from_pretrained(model_deployment_name, trust_remote_code=True)
                
                tokenizer_max_len_attr = getattr(temp_tokenizer_for_max_len, 'model_max_length', default_max_len)

                if not isinstance(tokenizer_max_len_attr, int) or tokenizer_max_len_attr <= 0 or tokenizer_max_len_attr > 2_000_000:
                    hlog(f"UTIL WARNING: Fallback tokenizer for {model_deployment_name} reported an unusual max length ({tokenizer_max_len_attr}). "
                         f"Using util default: {default_max_len}.")
                    model_max_len = default_max_len
                else:
                    model_max_len = tokenizer_max_len_attr
                    hlog(f"Util: Using model_max_length from AutoTokenizer.model_max_length: {model_max_len} for {model_deployment_name}")
            except ImportError:
                hlog(f"UTIL ERROR: transformers library not installed. Cannot use AutoTokenizer fallback for '{model_deployment_name}'. Using util default: {default_max_len}.")
                model_max_len = default_max_len
            except Exception as e_hf_tokenizer:
                hlog(f"UTIL ERROR: Fallback using AutoTokenizer for '{model_deployment_name}' also failed: {e_hf_tokenizer}. "
                     f"Using util default model_max_length: {default_max_len}.")
                model_max_len = default_max_len
            finally:
                if temp_tokenizer_for_max_len:
                    del temp_tokenizer_for_max_len
        
        if model_max_len <=0:
             hlog(f"UTIL WARNING: Determined model_max_length is non-positive ({model_max_len}) for {model_deployment_name}. Resetting to util default {default_max_len}.")
             model_max_len = default_max_len
        return model_max_len

    @staticmethod
    def create_generation_adapter_spec(original_adapter_spec: Any, generation_method_params: Dict[str, Any]) -> Any:
        """
        Creates a new AdapterSpec configured for generation, updating specified parameters.
        Ensures 'method' is set to 'generation'.
        """
        params_to_update = generation_method_params.copy()
        params_to_update.setdefault("method", "generation")

        fields_to_potentially_clear_or_set = [
            "instructions", "input_prefix", "output_prefix", "input_suffix", "output_suffix",
            "max_tokens", "temperature", "stop_sequences", "num_outputs", "random",
            "global_prefix", "global_suffix", "reference_prefix", "reference_suffix"
        ]
        for field in fields_to_potentially_clear_or_set:
            if field not in params_to_update:
                pass
        
        return replace(original_adapter_spec, **params_to_update)

    @staticmethod
    def check_prompt_length(
        prompt_text: str,
        model_name_for_tokenizer: str,
        tokenizer_service: TokenizerService,
        max_allowable_prompt_tokens: int
    ) -> Tuple[bool, int]:
        """
        Checks if the tokenized prompt_text fits within max_allowable_prompt_tokens.
        Returns a tuple: (is_valid_length, number_of_prompt_tokens).
        Returns (False, -1) if tokenization fails.
        """
        try:
            tokenization_request = TokenizationRequest(text=prompt_text, tokenizer=model_name_for_tokenizer)
            tokenization_result: TokenizationRequestResult = tokenizer_service.tokenize(tokenization_request)
            num_prompt_tokens = len(tokenization_result.tokens)
            return num_prompt_tokens <= max_allowable_prompt_tokens, num_prompt_tokens
        except Exception as e_tok_service:
            hlog(f"UTIL ERROR: TokenizerService failed for tokenizer '{model_name_for_tokenizer}'. Prompt: '{prompt_text[:100]}...'. Error: {e_tok_service}")
            return False, -1

    @staticmethod
    def format_helm_stats(
        calculated_metrics: Dict[str, float],
        strategy_metric_name_prefix: str,
        split: str = "test"
    ) -> List[Dict[str, Any]]:
        """
        Formats calculated metrics into the list of dictionaries expected by HELM.
        """
        final_helm_stats: List[Dict[str, Any]] = []
        for metric_name, metric_value in calculated_metrics.items():
            metric_value_rounded = np.round(metric_value, 2)

            final_helm_stats.append({
                "name": {"name": f"{strategy_metric_name_prefix} {metric_name})", "split": split}, 
                "count": 1, 
                "sum": metric_value_rounded,
                "sum_squared": np.round(metric_value_rounded**2, 2),
                "min": metric_value_rounded,
                "max": metric_value_rounded,
                "mean": metric_value_rounded, 
                "variance": 0.0, 
                "stddev": 0.0,
            })
        return final_helm_stats

    @staticmethod
    def get_spacy_tagger(language: str) -> Any:
        """
        Loads and returns a spaCy language model for POS tagging.
        Disables unnecessary components (parser, NER) for speed.
        """
        normalized_lang = language.lower().split('_')[0]
        model_name = UtilsContamination.SPACY_MODEL_MAP.get(normalized_lang)

        if not model_name:
            hlog(f"UTIL ERROR: No spaCy model mapped for language '{language}' (normalized to '{normalized_lang}').")
            raise ValueError(f"SpaCy model not configured for language: {language}")

        try:
            hlog(f"UTIL INFO: Attempting to load spaCy model: {model_name} for language {normalized_lang}")
            return spacy.load(model_name, disable=["parser", "ner"])
        except ImportError:
            hlog("UTIL ERROR: spaCy library not installed. Please install it: pip install spacy")
            raise
        except OSError:
            hlog(f"UTIL ERROR: spaCy model '{model_name}' not found. "
                 f"Please download it: python -m spacy download {model_name}")
            raise 
        except Exception as e:
            hlog(f"UTIL ERROR: An unexpected error occurred while loading spaCy model {model_name}: {e}")
            raise