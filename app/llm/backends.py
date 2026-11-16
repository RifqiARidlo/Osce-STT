"""Backend LLM lokal dengan constrained decoding berbasis JSON Schema.

- Ollama  : `format=<json schema>` (structured outputs, Ollama >= 0.5). Ollama berjalan sebagai proses
            terpisah dan mengelola VRAM sendiri; `keep_alive=0` membuat model langsung di-unload
            setelah respons -> setara memory swap.
- llama-cpp-python: `response_format={"type": "json_object", "schema": ...}` -> grammar GBNF internal.
Keduanya tidak butuh Outlines (mengurangi dependency).
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Protocol

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMBackend(Protocol):
    name: str

    def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> str: ...

    def close(self) -> None: ...


class OllamaBackend:
    def __init__(self, base_url: str, model: str, num_ctx: int, timeout_s: int, unload_after: bool,
                 options: dict[str, Any] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        # default deterministik untuk rangkuman; `options` menimpa (mis. sampling untuk generator data)
        self.options: dict[str, Any] = {"temperature": 0, "num_ctx": num_ctx, **(options or {})}
        self.model = model
        self.num_ctx = num_ctx
        self.timeout_s = timeout_s
        self.unload_after = unload_after
        self.name = f"ollama:{model}"

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:500]
            raise LLMError(f"Ollama HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LLMError(f"Ollama tidak dapat dihubungi di {self.base_url}: {exc}") from exc

    def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        data = self._post("/api/chat", {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "format": schema,
            "stream": False,
            "keep_alive": 0 if self.unload_after else "10m",
            "options": self.options,
        })
        try:
            return data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"Respons Ollama tidak terduga: {str(data)[:300]}") from exc

    def close(self) -> None:
        # paksa unload bila sebelumnya keep_alive > 0
        try:
            self._post("/api/generate", {"model": self.model, "keep_alive": 0})
        except LLMError:
            pass


class LlamaCppBackend:
    def __init__(self, model_path: str, n_gpu_layers: int, n_ctx: int) -> None:
        if not model_path:
            raise LLMError("APP_LLAMACPP_MODEL_PATH belum diisi (path file .gguf).")
        from llama_cpp import Llama

        self._llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, verbose=False)
        self.name = f"llamacpp:{model_path.rsplit('/', 1)[-1]}"

    def generate_json(self, system: str, user: str, schema: dict[str, Any]) -> str:
        out = self._llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={"type": "json_object", "schema": schema},
            temperature=0.0,
        )
        return out["choices"][0]["message"]["content"]

    def close(self) -> None:
        close = getattr(self._llm, "close", None)
        if callable(close):
            close()
        self._llm = None
