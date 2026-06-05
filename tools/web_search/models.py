"""Data models for web search results."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class RetrievedDocument:
    title: str
    url: str
    content: str
