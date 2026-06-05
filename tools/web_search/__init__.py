"""Web search package."""

from .models import RetrievedDocument, SearchResult
from .service import WebSearchService, WebSearchTool, search_web

__all__ = [
    "RetrievedDocument",
    "SearchResult",
    "WebSearchService",
    "WebSearchTool",
    "search_web",
]
