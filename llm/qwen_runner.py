"""Runner base para un modelo local tipo Qwen."""


class QwenRunner:
    """Implementación mínima para desacoplar el modelo del chat."""

    def __init__(self, model_name: str = "qwen-local") -> None:
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        return (
            f"[{self.model_name}] Respuesta simulada al mensaje: {prompt}"
        )
