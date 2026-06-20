import hashlib
import re
from typing import Optional, Tuple

from pingpoint.backend import detect_backend, Backend


def clean_ansi(text: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', text)


_preferred_backend: Optional[str] = None


def set_preferred_backend(name: Optional[str]) -> None:
    global _preferred_backend
    _preferred_backend = name


def _get_backend() -> Optional[Backend]:
    return detect_backend(prefer=_preferred_backend)


def call_generate(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    seed: Optional[int] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    timeout: int = 120,
) -> Optional[Tuple[str, float]]:
    backend = _get_backend()
    if backend is None:
        return None
    try:
        return backend.generate(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            top_p=top_p,
            top_k=top_k,
            timeout=timeout,
        )
    except Exception:
        return None


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
    return call_generate(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        top_p=top_p,
        top_k=top_k,
        timeout=timeout,
    )


def call_ollama(
    model: str,
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout: int = 120,
) -> Optional[Tuple[str, float]]:
    return call_generate(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def get_model_digest(model: str) -> Optional[str]:
    backend = _get_backend()
    if backend:
        return backend.model_digest(model)
    return None


def get_ollama_version() -> str:
    backend = _get_backend()
    if backend:
        return backend.version()
    return "unknown"


def get_ollama_binary_hash() -> Optional[str]:
    backend = _get_backend()
    if backend:
        return backend.binary_hash()
    return None
