"""Query expansion helpers for web search."""

from __future__ import annotations

import re

STOPWORDS = {
    "a",
    "actualmente",
    "an",
    "and",
    "are",
    "de",
    "el",
    "en",
    "es",
    "for",
    "how",
    "is",
    "esta",
    "la",
    "las",
    "los",
    "of",
    "on",
    "or",
    "que",
    "the",
    "to",
    "what",
    "who",
    "y",
}

TERM_TRANSLATIONS = {
    "actualmente": "current",
    "estado": "status",
    "estatus": "status",
    "hoy": "today",
    "lesion": "injury",
    "lesionada": "injury",
    "lesionado": "injury",
    "lesión": "injury",
    "lesionadas": "injuries",
    "lesionadoa": "injury",
    "lesionados": "injuries",
    "lesiones": "injuries",
    "lesiono": "injury",
    "quien": "who",
}


def build_search_queries(query: str) -> list[str]:
    queries: list[str] = []
    normalized_query = " ".join(query.split())
    if normalized_query:
        queries.append(normalized_query)

    entity_query = extract_entity_query(query)
    query_terms = extract_query_terms(query)
    translated_terms = [TERM_TRANSLATIONS.get(term, term) for term in query_terms]
    intent_terms = [term for term in extract_intent_terms(query) if term]

    if entity_query and intent_terms:
        joined_intent = " ".join(intent_terms)
        queries.append(f"{entity_query} {joined_intent}")
        queries.append(f'"{entity_query}" {joined_intent}')

    if translated_terms:
        queries.append(" ".join(translated_terms))

    if query_terms:
        queries.append(" ".join(query_terms))

    if entity_query:
        queries.append(entity_query)

    return _dedupe_queries(queries)


def extract_query_terms(query: str) -> list[str]:
    normalized_query = _normalize_text(query)
    tokens = re.findall(r"[a-z0-9]+", normalized_query)
    filtered_tokens = [
        token for token in tokens if len(token) >= 2 and token not in STOPWORDS
    ]
    return filtered_tokens or tokens


def extract_entity_query(query: str) -> str:
    entity_parts = re.findall(r"\b[A-Z][a-z]+\b", query)
    if len(entity_parts) >= 2:
        return " ".join(entity_parts)
    return ""


def extract_entity_terms(query: str) -> list[str]:
    entity_query = extract_entity_query(query)
    if not entity_query:
        return []
    return re.findall(r"[a-z0-9]+", entity_query.lower())


def extract_intent_terms(query: str) -> list[str]:
    entity_terms = set(extract_entity_terms(query))
    translated_terms = [TERM_TRANSLATIONS.get(term, term) for term in extract_query_terms(query)]
    return [term for term in translated_terms if term not in entity_terms]


def _dedupe_queries(queries: list[str]) -> list[str]:
    deduped_queries: list[str] = []
    seen: set[str] = set()
    for candidate in queries:
        normalized_candidate = candidate.strip()
        if not normalized_candidate:
            continue
        key = normalized_candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_queries.append(normalized_candidate)
    return deduped_queries


def _normalize_text(text: str) -> str:
    return (
        text.lower()
        .replace("á", "a")
        .replace("é", "e")
        .replace("í", "i")
        .replace("ó", "o")
        .replace("ú", "u")
    )
