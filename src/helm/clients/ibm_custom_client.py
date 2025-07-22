"""
Client for interacting with custom inference models (deployments) on 
IBM Cloud (Watsonx). This client makes direct REST API calls to the 
deployment endpoint, allowing full control over inference parameters.
"""
import requests
import time
from threading import Lock, Semaphore
from typing import Dict, Any, List

from helm.common.cache import CacheConfig
from helm.common.hierarchical_logger import hlog
from helm.common.request import (
    Request,
    RequestResult,
    Token,
    GeneratedOutput,
    wrap_request_time,
    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
)
from helm.clients.client import CachingClient

# Default limit for concurrent requests to the IBM API.
MAX_CONCURRENT_REQUESTS = 8
__semaphores: Dict[str, Semaphore] = dict()
__semaphores_lock = Lock()


def _get_semaphore(model: str, max_concurrent_requests: int) -> Semaphore:
    """Gets a semaphore to limit concurrency per model/deployment."""
    with __semaphores_lock:
        if model not in __semaphores:
            __semaphores[model] = Semaphore(max_concurrent_requests)
    return __semaphores[model]


class IbmCustomClient(CachingClient):
    """
    Client for custom models on IBM Watsonx via REST API.
    Handles authentication, caching, and dynamic payload construction for inference.
    """

    _IAM_URL = "https://iam.cloud.ibm.com/identity/token"
    _REGION_TO_URL = {
        "us-south": "https://us-south.ml.cloud.ibm.com",
        "eu-de": "https://eu-de.ml.cloud.ibm.com",
        "jp-tok": "https://jp-tok.ml.cloud.ibm.com",
        "au-syd": "https://au-syd.ml.cloud.ibm.com",
    }

    def __init__(
        self,
        cache_config: CacheConfig,
        api_key: str,
        region: str,
        project_id: str,
        deployment_id: str,
        api_version: str = "2023-05-29",
        max_concurrent_requests: int = MAX_CONCURRENT_REQUESTS,
        **kwargs,
    ):
        """
        Initializes the IbmCustomClient.

        Args:
            cache_config: The cache configuration.
            api_key: The IBM Cloud API key.
            region: The IBM Cloud region for the deployment.
            project_id: The Watsonx project ID.
            deployment_id: The specific ID of the custom model deployment.
            api_version: The version of the Watsonx API to use.
            max_concurrent_requests: The maximum number of parallel requests.
        """
        super().__init__(cache_config=cache_config)
        self.api_key = api_key
        self.project_id = project_id
        self.deployment_id = deployment_id
        self.max_concurrent_requests = max_concurrent_requests

        if region.lower() not in self._REGION_TO_URL:
            raise ValueError(f"Region '{region}' is not supported. Supported regions: {list(self._REGION_TO_URL.keys())}")
        
        base_url = self._REGION_TO_URL[region.lower()]
        self.endpoint_url = f"{base_url}/ml/v1/deployments/{self.deployment_id}/text/generation?version={api_version}"
        
        self._access_token: str | None = None
        self._token_expiry_time: float = 0
        self._token_lock = Lock()

        hlog.info(f"Started IBM Custom Client for deployment '{self.deployment_id}' in region '{region}'")
        if kwargs:
            hlog.info(f"Ignoring unexpected client arguments: {list(kwargs.keys())}")

    def _get_access_token(self) -> str:
        """Gets a Bearer access token from IBM Cloud IAM, with caching and automatic renewal."""
        with self._token_lock:
            if self._access_token and time.time() < self._token_expiry_time:
                return self._access_token

            hlog.info("Requesting new IBM IAM access token...")
            headers = {"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"}
            data = {"grant_type": "urn:ibm:params:oauth:grant-type:apikey", "apikey": self.api_key}
            
            response = requests.post(self._IAM_URL, headers=headers, data=data)
            response.raise_for_status()
            
            token_data = response.json()
            self._access_token = token_data["access_token"]
            self._token_expiry_time = time.time() + token_data.get("expires_in", 3600) - 60
            
            hlog.info("Successfully obtained new IBM IAM access token.")
            return self._access_token

    def make_request(self, request: Request) -> RequestResult:
        """
        Executes an inference request, dynamically building the payload
        based on the parameters from the HELM request.
        """
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        parameters: Dict[str, Any] = {
            "max_new_tokens": request.max_tokens,
            "min_new_tokens": 0,
            "stop_sequences": request.stop_sequences,
        }

        if request.temperature > 1e-7:
            parameters["decoding_method"] = "sample"
            parameters["temperature"] = request.temperature
            parameters["top_p"] = request.top_p
        else:
            parameters["decoding_method"] = "greedy"
        
        if request.frequency_penalty != 0:
            parameters["repetition_penalty"] = request.frequency_penalty

        payload = {
            "input": request.prompt,
            "parameters": parameters,
            "moderations": {
                "hap": {"input": {"enabled": True, "threshold": 0.5}, "output": {"enabled": True, "threshold": 0.5}},
            }
        }

        def do_it() -> Dict[str, Any]:
            """Encapsulated function that performs the network call."""
            access_token = self._get_access_token()
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {access_token}",
            }
            
            semaphore = _get_semaphore(self.deployment_id, self.max_concurrent_requests)
            with semaphore:
                response = requests.post(self.endpoint_url, headers=headers, json=payload)
            
            response.raise_for_status()
            return response.json()

        try:
            raw_request_for_cache = {"model": request.model, **payload}
            cache_key = CachingClient.make_cache_key(raw_request_for_cache, request)
            response_body, cached = self.cache.get(cache_key, wrap_request_time(do_it))

            completions = []
            if "results" in response_body and response_body["results"]:
                for res in response_body["results"]:
                    # The deployment API may not provide logprobs or individual tokens.
                    generated_text = res.get("generated_text", "")
                    completion = GeneratedOutput(text=generated_text, logprob=0, tokens=[])
                    completions.append(completion)
            
            return RequestResult(
                success=True,
                cached=cached,
                request_time=response_body["request_time"],
                request_datetime=response_body.get("request_datetime"),
                completions=completions,
                embedding=[],
            )
        except requests.exceptions.RequestException as e:
            error_msg = f"IBM Custom Client HTTP error: {e}"
            if e.response is not None:
                # Include the response body in the error log, crucial for debugging 400 errors.
                error_msg += f" | Status Code: {e.response.status_code} | Response: {e.response.text}"
            return RequestResult(success=False, cached=False, error=error_msg, completions=[], embedding=[])
        except Exception as e:
            error: str = f"IBM Custom Client failed with error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])