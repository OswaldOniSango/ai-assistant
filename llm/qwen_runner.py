"""Runner para un modelo local Qwen en formato GGUF."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError as exc:  # pragma: no cover - depende del entorno
    Llama = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
HOME_MODELS_DIR = Path.home() / "local-ai-workspace" / "models"


def _resolve_model_path() -> Path:
    env_path = os.getenv("QWEN_MODEL_PATH")
    if env_path:
        model_path = Path(env_path).expanduser()
        if model_path.exists():
            return model_path
        raise FileNotFoundError(
            f"QWEN_MODEL_PATH apunta a un archivo inexistente: {model_path}"
        )

    search_roots = (MODELS_DIR, HOME_MODELS_DIR)
    for root in search_roots:
        if not root.exists():
            continue

        direct_candidates = sorted(root.glob("*.gguf"))
        if direct_candidates:
            return direct_candidates[0]

        nested_candidates = sorted(root.rglob("*.gguf"))
        if nested_candidates:
            return nested_candidates[0]

    raise FileNotFoundError(
        "No se encontró un modelo GGUF. Define QWEN_MODEL_PATH, agrega un "
        "archivo .gguf dentro de local-ai-assistant/models/ o colócalo en "
        "~/local-ai-workspace/models/."
    )


@lru_cache(maxsize=1)
def _load_model() -> "Llama":
    if Llama is None:
        raise RuntimeError(
            "Falta instalar llama-cpp-python. "
            "Instálalo con: pip install llama-cpp-python"
        ) from IMPORT_ERROR

    model_path = _resolve_model_path()
    n_ctx = int(os.getenv("QWEN_CONTEXT_SIZE", "32768"))
    n_threads = int(os.getenv("QWEN_THREADS", str(os.cpu_count() or 4)))

    return Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=False,
    )


def ask_model(prompt: str) -> str:
    if not prompt.strip():
        raise ValueError("El prompt no puede estar vacío.")

    model = _load_model()

    try:
        response = model.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.2,
            max_tokens=256,
        )
        message = response["choices"][0]["message"]["content"]
        return message.strip()
    except Exception:
        fallback_prompt = f"User: {prompt}\nAssistant:"
        response = model.create_completion(
            prompt=fallback_prompt,
            temperature=0.2,
            max_tokens=256,
            stop=["User:", "\n\nUser:"],
        )
        text = response["choices"][0]["text"]
        return text.strip()


class QwenRunner:
    """Adaptador simple para desacoplar el chat del backend local."""

    def generate(self, prompt: str) -> str:
        return ask_model(prompt)
