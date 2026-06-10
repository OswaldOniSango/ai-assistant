"""Web search package."""

from .models import RetrievedDocument, SearchResult
from .service import WebSearchService, search_web

__all__ = [
    "RetrievedDocument",
    "SearchResult",
    "WebSearchService",
    "search_web",
]
