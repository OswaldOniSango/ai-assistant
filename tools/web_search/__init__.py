"""Web search package."""

from .models import SearchResult
from .service import WebSearchService, WebSearchTool, search_web

__all__ = ["SearchResult", "WebSearchService", "WebSearchTool", "search_web"]
