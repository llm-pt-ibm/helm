import json
import logging
import requests

from dataclasses import asdict
from typing import Any, Dict, List
from helm.common.cache import CacheConfig
from helm.common.general import get_credentials
from helm.common.request import (
    wrap_request_time,
    Request,
    RequestResult,
    GeneratedOutput,
    Token,
    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
)
from helm.clients.client import CachingClient
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class Power9Client(CachingClient):
    """
    Implements a HELM client for an API running on an IBM Power 9 machine, with endpoints to load the model and generate text.
    The credentials (api_key, hf_token, base_url) are loaded from the *credentials.conf* file, that should be created inside prod-env.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        timeout: int = 300,
        do_cache: bool = False,
        pretrained_model_name_or_path: str = "ibm-granite/granite-3.3-8b-instruct", # default model is granite-3.3-8b-instruct
    ):
        super().__init__(cache_config=cache_config)

        try:
            credentials = get_credentials()

            self.base_url = credentials.get("ibm-graniteBaseUrl")
            if not self.base_url:
                raise ValueError("Credential 'base_url' is missing in section 'power9_api' of credentials.conf.")

            self.api_key = credentials.get("ibm-graniteApiKey")
            if not self.api_key:
                raise ValueError("Credential 'api_key' is missing in section 'power9_api' of credentials.conf.")

            self.hf_token = credentials.get("ibm-graniteHfToken")
            if not self.hf_token:
                raise ValueError("Credential 'hf_token' (HuggingFace token) is missing in section 'power9_api' of credentials.conf.")


        except Exception as e:
            logger.critical(f"Fatal error while initializing Power9Client: {e}", exc_info=True)
            raise

        self.timeout = timeout
        self.do_cache = do_cache
        self.model_name = pretrained_model_name_or_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _load_model_on_init(self):
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }
        payload = {
            "model_name": self.model_name,
            "hf_token": self.hf_token,
        }

        try:
            url = f"{self.base_url}/load_model"
            logger.info(f"Attempting to load model '{self.model_name}' at '{url}'…")
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            logger.info(f"Model '{self.model_name}' loaded successfully: {response.json()}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Error loading model '{self.model_name}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading model '{self.model_name}': {e}")
            raise

    def make_request(self, request: Request) -> RequestResult:
        cache_key = asdict(request)

        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }

        raw_request = {
            "prompt": request.prompt,
            "model_name": self.model_name,
            "hf_token": self.hf_token,
        }

        try:
            def send_request() -> Dict[str, Any]:
                url = f"{self.base_url}/generate"
                print(url)
                logger.debug(f"Sending request to {url} with payload: {raw_request}")
                print("cheguei aqui")
                response = requests.post(url, json=raw_request, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                return response.json()

            if self.do_cache:
                response_data, cached = self.cache.get(cache_key, wrap_request_time(send_request))
            else:
                response_data, cached = send_request(), False
            
            result_data = response_data.get("result", {})
            generated_text = result_data.get("text", "")
            log_probs = result_data.get("log_probs", [])

            # first, the text is tokenized by the tokenizer
            # it is important that the tokenizer used here is the same as the one used in the model
            tokenized_text = self.tokenizer.tokenize(generated_text)
            
            tokens: List[Token] = [
                Token(text=token, logprob=log_prob)
                for token, log_prob in zip(tokenized_text, log_probs)
            ]
            total_logprob = sum(log_probs)

            completions = [
                GeneratedOutput(
                    text=generated_text,
                    logprob=total_logprob,
                    tokens=tokens,
                )
            ]

            return RequestResult(
                success=True,
                cached=cached,
                error=None,
                completions=completions,
                embedding=[],
                request_time=response_data.get("request_time", 0.0),
            )

        except requests.exceptions.RequestException as e:
            error = f"HTTP request error to Power 9 API: {e}"
            logger.error(error, exc_info=True)
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        except json.JSONDecodeError as e:
            error = f"Error parsing JSON response from API: {e}"
            logger.error(error, exc_info=True)
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        except Exception as e:
            error = f"Unexpected error in Power9Client: {e}"
            logger.critical(error, exc_info=True)
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])