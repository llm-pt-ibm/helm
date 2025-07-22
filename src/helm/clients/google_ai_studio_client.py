"""
Client for Google AI Studio (Gemini) models using the `google-generativeai` SDK.
"""

import time
from typing import Any, Dict, List, Optional, cast
from threading import Semaphore, Lock

from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.optional_dependencies import handle_module_not_found_error
from helm.common.request import (
    GeneratedOutput,
    Request,
    RequestResult,
    Token,
    wrap_request_time,
    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
)
from helm.common.tokenization_request import (
    TokenizationRequest,
    TokenizationRequestResult,
)
from helm.tokenizers.tokenizer import Tokenizer
from helm.clients.client import CachingClient, truncate_sequence

try:
    import google.generativeai as genai
    from google.api_core import exceptions as google_exceptions
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["google-generativeai"])


# Concurrency control for the Google API.
# The free tier for Gemini has a requests per minute (RPM) limit (e.g., 15 for 1.5 Flash).
# We reduce concurrency to 1 to serialize requests and ensure the limit is not exceeded.
MAX_CONCURRENT_REQUESTS: int = 1
_semaphores: Dict[str, Semaphore] = {}
_semaphores_lock = Lock()


def _get_semaphore(model_name: str) -> Semaphore:
    """Returns a semaphore for a given model name to limit parallel requests."""
    with _semaphores_lock:
        if model_name not in _semaphores:
            _semaphores[model_name] = Semaphore(MAX_CONCURRENT_REQUESTS)
    return _semaphores[model_name]


class GoogleAIStudioClient(CachingClient):
    """
    A client for interacting with Google AI (Gemini) models via the `google-generativeai` SDK.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
    ):
        """
        Initializes the GoogleAIStudioClient.

        Args:
            tokenizer: The HELM tokenizer.
            tokenizer_name: The name of the HELM tokenizer.
            cache_config: The cache configuration.
            api_key: The Google API key.
        """
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name

        if not api_key:
            raise ValueError("A Google API key (`api_key`) is required.")

        genai.configure(api_key=api_key)
        hlog.info("Google AI Studio (Gemini) Client initialized.")

    def make_request(self, request: Request) -> RequestResult:
        """
        Main entry point for all requests to the Google API.

        This method checks for embedding requests and dispatches to the
        appropriate handler for completion requests.
        """
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        return self._make_completion_request(request)

    def _make_completion_request(self, request: Request) -> RequestResult:
        """
        Handles completion requests for Gemini models.
        """
        # The Google SDK uses a configuration object for generation parameters.
        generation_config = {
            "candidate_count": request.num_completions,
            "max_output_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            # top_k is also available but not a direct parameter in the HELM Request object.
        }

        # Add stop_sequences if they exist, as the SDK may not accept an empty list.
        if request.stop_sequences:
            generation_config["stop_sequences"] = request.stop_sequences

        # The `raw_request` is used to create the cache key. It must contain all parameters
        # that affect the output to ensure the cache works correctly.
        raw_request: Dict[str, Any] = {
            "model": request.model_engine,
            "contents": request.prompt,
            "generation_config": generation_config,
        }

        def do_it() -> Dict[str, Any]:
            """
            The actual function that makes the API call, managed by the caching system.
            """
            model = genai.GenerativeModel(request.model_engine)
            semaphore = _get_semaphore(request.model)
            with semaphore:
                response = model.generate_content(
                    contents=request.prompt, generation_config=generation_config
                )
                # A hardcoded delay to respect the free tier's RPM limits.
                time.sleep(1)
            # The Google response object is not directly JSON-serializable.
            # We convert it to a plain dictionary so it can be cached.
            return self._response_to_dict(response)

        try:
            cache_key = CachingClient.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except google_exceptions.GoogleAPICallError as e:
            error: str = f"Google API error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])
        except Exception as e:
            error: str = f"Unexpected error calling the Google API: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        # Process the response dictionary (from the cache or the API call).
        completions: List[GeneratedOutput] = []
        for raw_completion in response["candidates"]:
            # The generated text is within `parts[0]`.
            text = raw_completion["content"]["parts"][0]["text"]
            
            # Since the API does not return tokens or logprobs, we tokenize the text ourselves
            # to conform to the `GeneratedOutput` structure.
            tokenization_result: TokenizationRequestResult = self.tokenizer.tokenize(
                TokenizationRequest(text, tokenizer=self.tokenizer_name)
            )
            tokens: List[Token] = [
                Token(text=cast(str, raw_token), logprob=0) for raw_token in tokenization_result.raw_tokens
            ]
            
            completion = GeneratedOutput(
                text=text,
                logprob=0,
                tokens=tokens,
                finish_reason={"reason": raw_completion.get("finish_reason", "FINISH_REASON_UNSPECIFIED")},
            )
            completions.append(truncate_sequence(completion, request))

        return RequestResult(
            success=True,
            cached=cached,
            request_time=response["request_time"],
            request_datetime=response.get("request_datetime"),
            completions=completions,
            embedding=[],
        )

    def _response_to_dict(self, response) -> Dict[str, Any]:
        """
        Converts the `GenerateContentResponse` object from the Google SDK into a
        simple, serializable dictionary for caching.
        """
        candidates_list = []
        for candidate in response.candidates:
            # candidate.finish_reason is an Enum, so we get its string name.
            finish_reason_str = candidate.finish_reason.name
            
            parts_list = []
            for part in candidate.content.parts:
                parts_list.append({"text": part.text})

            candidates_list.append({
                "content": {"parts": parts_list},
                "finish_reason": finish_reason_str,
            })
        
        return {"candidates": candidates_list}