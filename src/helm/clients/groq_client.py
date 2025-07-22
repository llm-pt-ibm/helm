"""
Client for the Groq API, which provides high-speed inference for language models.
"""

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
    from groq import Groq, GroqError
    from httpx import Timeout  # Required for custom timeout configuration
except ModuleNotFoundError as e:
    handle_module_not_found_error(e, ["groq"])


# Serialize requests to stay within API rate limits.
MAX_CONCURRENT_REQUESTS: int = 1
_semaphores: Dict[str, Semaphore] = {}
_semaphores_lock = Lock()


def _get_semaphore(model_name: str) -> Semaphore:
    """Returns a semaphore for a given model name to limit parallel requests."""
    with _semaphores_lock:
        if model_name not in _semaphores:
            _semaphores[model_name] = Semaphore(MAX_CONCURRENT_REQUESTS)
    return _semaphores[model_name]


class GroqClient(CachingClient):
    """
    Client for interacting with models from the Groq API.
    The Groq API is OpenAI-compatible, which simplifies integration.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        tokenizer_name: str,
        cache_config: CacheConfig,
        api_key: Optional[str] = None,
    ):
        """
        Initializes the GroqClient.

        Args:
            tokenizer: The HELM tokenizer.
            tokenizer_name: The name of the HELM tokenizer.
            cache_config: The cache configuration.
            api_key: The Groq API key.
        """
        super().__init__(cache_config=cache_config)
        self.tokenizer = tokenizer
        self.tokenizer_name = tokenizer_name

        if not api_key:
            raise ValueError("A Groq API key (`api_key`) is required.")

        self.client = Groq(api_key=api_key, timeout=Timeout(120.0, connect=10.0))
        hlog.info("Groq client initialized.")

    def make_request(self, request: Request) -> RequestResult:
        """
        Main entry point for all requests to the Groq API.
        Currently, only chat completion is supported.
        """
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        return self._make_chat_request(request)

    def _make_chat_request(self, request: Request) -> RequestResult:
        """Handles chat completion requests for the Groq API."""
        messages: List[Dict[str, str]]
        if request.messages:
            messages = [{"role": msg["role"], "content": msg["content"]} for msg in request.messages]
        else:
            messages = [{"role": "user", "content": request.prompt}]

        raw_request: Dict[str, Any] = {
            "model": request.model_engine,
            "messages": messages,
            "temperature": 1e-7 if request.temperature == 0 else request.temperature,
            "top_p": request.top_p,
            "n": request.num_completions,
            "max_tokens": request.max_tokens,
            "stop": request.stop_sequences or None,  # API expects `None`, not an empty list.
            "stream": False,
        }

        def do_it() -> Dict[str, Any]:
            """The actual function that makes the API call, wrapped by the cache manager."""
            semaphore = _get_semaphore(request.model)
            with semaphore:
                response = self.client.chat.completions.create(**raw_request)
            # Use .model_dump() to get a serializable dictionary for caching.
            return response.model_dump(mode="json")

        try:
            cache_key = CachingClient.make_cache_key(raw_request, request)
            response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
        except GroqError as e:
            error: str = f"Groq API error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])
        except Exception as e:
            error: str = f"Unexpected error when calling the Groq API: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

        completions: List[GeneratedOutput] = []
        for raw_completion in response["choices"]:
            # The Groq API does not return tokens or logprobs in standard chat mode,
            # so we must tokenize the output text ourselves to conform to the HELM format.
            text = raw_completion["message"]["content"]
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
                finish_reason={"reason": raw_completion["finish_reason"]},
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