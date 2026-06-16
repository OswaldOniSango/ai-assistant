"""Runner for a local Qwen GGUF model."""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from pathlib import Path

try:
    from llama_cpp import Llama
except ImportError as exc:  # pragma: no cover - depends on the local environment
    Llama = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
HOME_MODELS_DIR = Path.home() / "local-ai-workspace" / "models"
DEFAULT_MAX_TOKENS = 512


def _resolve_model_path() -> Path:
    env_path = os.getenv("QWEN_MODEL_PATH")
    if env_path:
        model_path = Path(env_path).expanduser()
        if model_path.exists():
            return model_path
        raise FileNotFoundError(
            f"QWEN_MODEL_PATH points to a missing file: {model_path}"
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
        "No GGUF model was found. Set QWEN_MODEL_PATH, add a .gguf file under "
        "local-ai-assistant/models/, or place it under ~/local-ai-workspace/models/."
    )


@lru_cache(maxsize=1)
def _load_model() -> "Llama":
    if Llama is None:
        raise RuntimeError(
            "llama-cpp-python is not installed. "
            "Install it with: pip install llama-cpp-python"
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
        raise ValueError("Prompt cannot be empty.")

    model = _load_model()
    max_tokens = _max_tokens()
    final_prompt = build_direct_answer_prompt(prompt)

    try:
        response = model.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": final_prompt,
                }
            ],
            temperature=0.2,
            max_tokens=max_tokens,
        )
        message = response["choices"][0]["message"]["content"]
        return message.strip()
    except Exception:
        response = model.create_completion(
            prompt=final_prompt,
            temperature=0.2,
            max_tokens=max_tokens,
            stop=["User question:", "\n\nUser question:"],
        )
        text = response["choices"][0]["text"]
        return text.strip()


def _max_tokens() -> int:
    raw_value = os.getenv("QWEN_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))
    try:
        max_tokens = int(raw_value)
    except ValueError as exc:
        raise ValueError("QWEN_MAX_TOKENS must be an integer.") from exc

    if max_tokens < 1:
        raise ValueError("QWEN_MAX_TOKENS must be greater than zero.")

    return max_tokens


def build_direct_answer_prompt(user_prompt: str) -> str:
    return (
        "You are a local AI assistant.\n"
        f"Today's date is {date.today().isoformat()}. Your training data is "
        "older than this date.\n"
        "First identify the language of the user's question. Do not mention this analysis.\n"
        "Reply entirely in the same language as the user's question.\n"
        "The language of any supporting context must not change the reply language.\n"
        "If the question is in English, answer in English.\n"
        "If the question is in Spanish, answer in Spanish.\n"
        "If the answer depends on current events, live data, or anything that "
        "may have changed recently, do not guess: say you cannot verify it "
        "right now.\n"
        "Never invent facts, names, scores, or dates.\n"
        "Keep the answer clear and concise.\n\n"
        f"User question: {user_prompt}\n\n"
        "Answer:"
    )
