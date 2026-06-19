import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple

from pingpoint.models import Solution, SolutionMetadata


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def clean_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)


def call_ollama_api(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    timeout: int = 120,
) -> Optional[Tuple[str, float]]:
    url = f"{OLLAMA_HOST}/api/generate"
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
        output = clean_ansi(result.get("response", "").strip())
        if not output:
            return None
        return output, elapsed
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return None


def call_ollama(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 120,
) -> Optional[Tuple[str, float]]:
    return call_ollama_api(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def get_model_digest(model: str) -> Optional[str]:
    try:
        url = f"{OLLAMA_HOST}/api/show"
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


def get_ollama_version() -> str:
    try:
        url = f"{OLLAMA_HOST}/api/version"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        return data.get("version", "unknown")
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return "unknown"


def get_ollama_binary_hash() -> Optional[str]:
    if os.name == "nt":
        paths = [r"C:\Program Files\Ollama\ollama.exe", r"C:\Program Files (x86)\Ollama\ollama.exe"]
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
