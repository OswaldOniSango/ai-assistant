"""Generate search queries with the local model."""

from __future__ import annotations

from llm.qwen_runner import ask_model
from tools.web_search.query_builder import finalize_queries, parse_query_lines


def generate_search_queries(question: str, limit: int = 5) -> list[str]:
    if not question.strip():
        raise ValueError("La pregunta no puede estar vacía.")

    planner_prompt = (
        "You are helping a local AI assistant search the web.\n"
        "Generate 3 to 5 concise search engine queries that maximize retrieval quality.\n"
        "Infer the domain from the question.\n"
        "For technical questions, include specific domain terminology, likely keywords from documentation/articles, and avoid generic wording.\n"
        "Prefer web-friendly wording and keep named entities exact.\n"
        "If the user's question is in Spanish, you may output queries in English when that improves web search quality.\n"
        "Return only the queries, one per line, with no numbering, no bullets, and no explanation.\n\n"
        "Examples:\n"
        "User question: como puedo optimizar una query sql\n"
        "sql query optimization best practices\n"
        "how to optimize sql queries performance\n"
        "sql query tuning indexing explain analyze\n"
        "postgres query optimization guide\n\n"
        "User question: snowflake esta perdiendo clientes?\n"
        "Snowflake customer churn 2026\n"
        "Snowflake customer count quarterly results 2026\n"
        "Snowflake revenue growth customers 2026\n"
        "analyst report Snowflake demand trends 2026\n\n"
        f"User question: {question}\n"
    )
    raw_output = ask_model(planner_prompt)
    parsed_queries = parse_query_lines(raw_output)
    return finalize_queries(parsed_queries, fallback_query=question, limit=limit)
