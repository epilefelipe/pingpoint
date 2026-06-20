import hashlib
import json
import os
import time
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple


DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_OMLX_HOST = os.environ.get("OMLX_HOST", "http://localhost:8000")


class Backend(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        timeout: int = 120,
    ) -> Optional[Tuple[str, float]]:
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        ...

    @abstractmethod
    def version(self) -> str:
        ...

    def model_digest(self, model: str) -> Optional[str]:
        return None

    def binary_hash(self) -> Optional[str]:
        return None


class OllamaBackend(Backend):
    def __init__(self, host: str = DEFAULT_OLLAMA_HOST):
        self.host = host

    @property
    def name(self) -> str:
        return "ollama"

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        timeout: int = 120,
    ) -> Optional[Tuple[str, float]]:
        url = f"{self.host}/api/generate"
        body = {
            "model": model,
            "prompt": prompt,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "stream": False,
        }
        if seed is not None:
            body["options"]["seed"] = seed
        if top_p is not None:
            body["options"]["top_p"] = top_p
        if top_k is not None:
            body["options"]["top_k"] = top_k

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            start = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode())
            elapsed = time.time() - start
            output = self._clean(result.get("response", "").strip())
            if not output:
                return None
            return output, elapsed
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return None

    def list_models(self) -> list[str]:
        try:
            url = f"{self.host}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return []

    def version(self) -> str:
        try:
            url = f"{self.host}/api/version"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            return data.get("version", "unknown")
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return "unknown"

    def model_digest(self, model: str) -> Optional[str]:
        try:
            url = f"{self.host}/api/show"
            body = json.dumps({"model": model}).encode()
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            digest = data.get("digest")
            if digest:
                return digest.split(":")[-1][:16]
            modified_at = data.get("modified_at", "")
            return hashlib.sha256(modified_at.encode()).hexdigest()[:16]
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return None

    def binary_hash(self) -> Optional[str]:
        if os.name == "nt":
            paths = [
                r"C:\Program Files\Ollama\ollama.exe",
                r"C:\Program Files (x86)\Ollama\ollama.exe",
            ]
        else:
            paths = ["/usr/bin/ollama", "/usr/local/bin/ollama"]
        for p in paths:
            f = Path(p)
            if f.exists():
                try:
                    data = f.read_bytes()
                    return hashlib.sha256(data).hexdigest()[:16]
                except (OSError, PermissionError):
                    pass
        return None

    @staticmethod
    def _clean(text: str) -> str:
        import re
        return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)

    @classmethod
    def probe(cls, host: str = DEFAULT_OLLAMA_HOST) -> bool:
        try:
            url = f"{host}/api/version"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
            return "version" in data
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return False


class OMLXBackend(Backend):
    def __init__(self, host: str = DEFAULT_OMLX_HOST):
        self.host = host

    @property
    def name(self) -> str:
        return "omlx"

    def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        seed: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        timeout: int = 120,
    ) -> Optional[Tuple[str, float]]:
        url = f"{self.host}/v1/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if seed is not None:
            body["seed"] = seed
        if top_p is not None:
            body["top_p"] = top_p

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            start = time.time()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode())
            elapsed = time.time() - start
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            output = content.strip()
            if not output:
                return None
            return output, elapsed
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return None

    def list_models(self) -> list[str]:
        try:
            url = f"{self.host}/v1/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            return [m["id"] for m in data.get("data", [])]
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return []

    def version(self) -> str:
        try:
            url = f"{self.host}/admin/info"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                info = json.loads(resp.read().decode())
            return info.get("version", "unknown")
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return "unknown"

    @classmethod
    def probe(cls, host: str = DEFAULT_OMLX_HOST) -> bool:
        try:
            url = f"{host}/v1/models"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode())
            return "data" in data
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            return False


BACKENDS: dict[str, type[Backend]] = {
    "ollama": OllamaBackend,
    "omlx": OMLXBackend,
}


def detect_backend(prefer: Optional[str] = None) -> Optional[Backend]:
    if prefer and prefer in BACKENDS:
        if BACKENDS[prefer].probe():
            return BACKENDS[prefer]()

    for name, cls in BACKENDS.items():
        if name == prefer:
            continue
        if cls.probe():
            return cls()

    return None


def get_backend(name: str) -> Optional[Backend]:
    cls = BACKENDS.get(name)
    if cls:
        return cls()
    return None
