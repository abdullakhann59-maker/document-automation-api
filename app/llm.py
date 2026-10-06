"""Direct Ollama REST calls: generation with JSON parsing and retries, and embeddings."""
import json
import logging
import re
import threading
from typing import Callable, TypeVar

import httpx

from .config import Settings
from .errors import AppError

log = logging.getLogger(__name__)
T = TypeVar("T")


def parse_json(raw: str) -> dict:
    """Pull a JSON object out of model output that may have fences or stray text around it."""
    text = re.sub(r"```(?:json)?", "", raw).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the output")
    text = text[start : end + 1]
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = json.loads(re.sub(r",\s*([}\]])", r"\1", text))  # trailing commas
    if not isinstance(obj, dict):
        raise ValueError("output is not a JSON object")
    return obj


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._http = httpx.Client(
            base_url=settings.ollama_url,
            timeout=httpx.Timeout(settings.llm_timeout_sec, connect=5.0),
        )
        self._sem = threading.Semaphore(1)  # one generation at a time on CPU

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        try:
            r = self._http.request(method, path, json=payload)
            r.raise_for_status()
            return r.json()
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise AppError(503, f"Ollama is not reachable at {self.settings.ollama_url}") from e
        except httpx.TimeoutException as e:
            raise AppError(504, f"Ollama timed out after {self.settings.llm_timeout_sec}s") from e
        except httpx.HTTPStatusError as e:
            raise AppError(503, f"Ollama error {e.response.status_code}: {e.response.text[:300]}") from e
        except httpx.HTTPError as e:
            raise AppError(503, f"Ollama request failed: {e}") from e

    def list_models(self) -> list[str]:
        return [m["name"] for m in self._request("GET", "/api/tags").get("models", [])]

    def has_model(self, name: str) -> bool:
        models = self.list_models()
        return name in models or f"{name}:latest" in models

    def generate(self, prompt: str, json_mode: bool = True, num_predict: int | None = None) -> str:
        s = self.settings
        payload = {
            "model": s.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": s.llm_temperature,
                "num_ctx": s.llm_num_ctx,
                "num_predict": num_predict or s.llm_num_predict,
            },
        }
        if json_mode:
            payload["format"] = "json"
        with self._sem:
            return self._request("POST", "/api/generate", payload).get("response", "")

    def generate_json(self, prompt: str, validate: Callable[[dict], T], num_predict: int | None = None) -> T:
        """Generate, parse and validate; on failure retry with the error appended."""
        attempt_prompt, raw, error = prompt, "", ""
        for attempt in range(self.settings.llm_max_retries + 1):
            raw = self.generate(attempt_prompt, num_predict=num_predict)
            try:
                return validate(parse_json(raw))
            except Exception as e:  # JSON or schema failure
                error = str(e)[:500]
                log.warning("LLM output invalid (attempt %d): %s", attempt + 1, error)
                attempt_prompt = (
                    f"{prompt}\n\nYour previous answer was invalid: {error}\n"
                    "Answer again with one valid JSON object only."
                )
        raise AppError(500, f"The model did not return valid JSON: {error}", raw_output=raw[:4000])

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        data = self._request("POST", "/api/embed", {"model": self.settings.embed_model, "input": texts})
        embeddings = data.get("embeddings") or []
        if len(embeddings) != len(texts):
            raise AppError(503, "Ollama returned the wrong number of embeddings")
        return embeddings
