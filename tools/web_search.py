"""Herramienta de búsqueda web basada en DuckDuckGo."""

from __future__ import annotations

import json
import re
import ssl
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen
import warnings

try:
    from duckduckgo_search import DDGS
except ImportError as exc:  # pragma: no cover - depende del entorno
    DDGS = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


def search_web(query: str) -> list[dict[str, str]]:
    """Busca resultados web y devuelve título, URL y snippet."""
    if not query.strip():
        raise ValueError("La consulta no puede estar vacía.")

    if DDGS is None:
        raise RuntimeError(
            "Falta instalar duckduckgo-search. "
            "Instálalo con: pip install duckduckgo-search"
        ) from IMPORT_ERROR

    fallback_results = _search_with_instant_answer(query)
    if fallback_results:
        return [_normalize_result(result) for result in fallback_results]

    raw_results = _search_with_ddgs(query)
    return [_normalize_result(result) for result in raw_results]


def _normalize_result(result: dict[str, Any]) -> dict[str, str]:
    return {
        "title": str(result.get("title", result.get("Heading", ""))).strip(),
        "url": str(
            result.get("href", result.get("url", result.get("FirstURL", "")))
        ).strip(),
        "snippet": str(
            result.get("body", result.get("snippet", result.get("Text", "")))
        ).strip(),
    }


class WebSearchTool:
    """Adaptador simple para desacoplar la capa de herramientas."""

    def search(self, query: str) -> list[dict[str, str]]:
        return search_web(query)


def _search_with_ddgs(query: str) -> list[dict[str, Any]]:
    try:
        with warnings.catch_warnings(record=True):
            with DDGS() as ddgs:
                return ddgs.text(query, max_results=5)
    except Exception:
        return []


def _search_with_instant_answer(query: str) -> list[dict[str, str]]:
    params = urlencode(
        {
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
        }
    )
    url = f"https://api.duckduckgo.com/?{params}"
    ssl_context = ssl._create_unverified_context()

    try:
        with urlopen(url, timeout=10, context=ssl_context) as response:
            payload = json.load(response)
    except Exception:
        return []

    results: list[dict[str, str]] = []

    if payload.get("AbstractText"):
        results.append(
            {
                "title": str(payload.get("Heading", "")).strip(),
                "url": str(payload.get("AbstractURL", "")).strip(),
                "snippet": str(payload.get("AbstractText", "")).strip(),
            }
        )

    for topic in payload.get("RelatedTopics", []):
        _collect_related_topics(topic, results)
        if len(results) >= 5:
            break

    return results[:5]


def _collect_related_topics(
    topic: dict[str, Any], results: list[dict[str, str]]
) -> None:
    nested_topics = topic.get("Topics")
    if isinstance(nested_topics, list):
        for nested_topic in nested_topics:
            _collect_related_topics(nested_topic, results)
            if len(results) >= 5:
                return
        return

    text = str(topic.get("Text", "")).strip()
    url = str(topic.get("FirstURL", "")).strip()
    if not text or not url:
        return

    title = text.split(" - ", 1)[0].strip()
    snippet = re.sub(r"^.*? - ", "", text, count=1).strip()
    results.append(
        {
            "title": title,
            "url": url,
            "snippet": snippet or text,
        }
    )
