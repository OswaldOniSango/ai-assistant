"""Herramienta base para búsquedas web."""


class WebSearchTool:
    """Stub inicial para desacoplar la capa de herramientas."""

    def search(self, query: str) -> str:
        if not query:
            return "No se recibió una consulta para buscar."

        return f"Búsqueda simulada para: {query}"
