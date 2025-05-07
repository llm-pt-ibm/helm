import time
import subprocess
from dataclasses import asdict
from typing import Any, Dict, Optional, List
from helm.common.cache import CacheConfig
from helm.common.request import (
    wrap_request_time,
    Request,
    RequestResult,
    GeneratedOutput,
    Token,
    EMBEDDING_UNAVAILABLE_REQUEST_RESULT,
)
from helm.clients.client import CachingClient
import requests


class DirectSSHDeepSeekClient(CachingClient):
    """
    Client for connecting to a server via direct SSH tunnel
    and making DeepSeek inference requests over HTTP.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        server_ip: str = "192.168.189.103",
        local_port: int = 8080,
        remote_port: int = 8080,
        timeout: int = 3000,
        do_cache: bool = False,
    ):
        super().__init__(cache_config=cache_config)
        self.server_ip = server_ip
        self.local_port = local_port
        self.remote_port = remote_port
        self.timeout = timeout
        self.do_cache = do_cache
        self.base_url = f"http://localhost:{local_port}"
        # Process handler for SSH connection
        self.ssh_process = None
        # Connect on initialization
        self._establish_connection()

    def make_request(self, request: Request) -> RequestResult:
        cache_key = asdict(request)
        if request.embedding:
            return EMBEDDING_UNAVAILABLE_REQUEST_RESULT

        raw_request = {"prompt": request.prompt, "n_predict": 128, "return_tokens": "true", "n_probs": 1}
        try:

            def do_it() -> Dict[str, Any]:
                url = f"{self.base_url}/completion"
                start_time = time.time()
                response = requests.post(
                    url, json=raw_request, timeout=self.timeout, headers={"Content-Type": "application/json"}
                )
                request_time = time.time() - start_time
                response.raise_for_status()
                response_data = response.json()
                response_data["request_time"] = request_time
                return response_data

            if self.do_cache:
                response, cached = self.cache.get(cache_key, wrap_request_time(do_it))
            else:
                response, cached = do_it(), False

            tokens = [
                Token(text=token["content"], logprob=token["probs"][0]["prob"])
                for token in response["completion_probabilities"]
            ]
            logprob = sum([token["probs"][0]["prob"] for token in response["completion_probabilities"]])
            completions = [GeneratedOutput(text=response["content"], tokens=tokens, logprob=logprob)]

            return RequestResult(
                success=True,
                cached=cached,
                error=None,
                completions=completions,
                embedding=[],
                request_time=response.get("request_time", 0),
            )
        except requests.exceptions.RequestException as e:
            error = f"Request error: {e}"
            return RequestResult(success=False, cached=False, error=error, completions=[], embedding=[])

    def __del__(self):
        """Clean up connections when object is destroyed."""
        self._cleanup_connections()

    def _establish_connection(self) -> None:
        """Establish SSH connection to the server."""
        try:
            # Create SSH tunnel
            self._create_ssh_tunnel()
            # Wait for SSH tunnel to establish
            time.sleep(2)
            # Verify connection
            self._verify_connection()
        except Exception as e:
            self._cleanup_connections()
            raise ConnectionError(f"Failed to establish connection to server: {str(e)}")

    def _create_ssh_tunnel(self) -> None:
        """Create SSH tunnel to server."""
        ssh_cmd = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-L",
            f"{self.local_port}:localhost:{self.remote_port}",
            f"root@{self.server_ip}",
        ]
        print(f"Establishing SSH tunnel with command: {' '.join(ssh_cmd)}")
        self.ssh_process = subprocess.Popen(
            ssh_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # Check if SSH process started successfully
        if self.ssh_process.poll() is not None:
            stderr = self.ssh_process.stderr.read().decode("utf-8")
            raise ConnectionError(f"SSH tunnel failed to start: {stderr}")

    def _verify_connection(self) -> None:
        """Verify connection to the DeepSeek server."""
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            if response.status_code != 200:
                print(f"Warning: Connection verification returned status code {response.status_code}")
        except requests.exceptions.RequestException:
            print("Warning: Could not verify connection. Continuing anyway...")

    def _cleanup_connections(self) -> None:
        """Clean up SSH connection."""
        if self.ssh_process:
            print("Terminating SSH tunnel")
            self.ssh_process.terminate()
            self.ssh_process = None
