"""Generate search queries with the local model."""

from __future__ import annotations

from datetime import date
from typing import Callable

from llm.qwen_runner import ask_model
from tools.web_search.query_builder import finalize_queries, parse_query_lines


def generate_search_queries(
    question: str,
    limit: int = 5,
    model: Callable[[str], str] = ask_model,
) -> list[str]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    today = date.today().isoformat()
    planner_prompt = (
        "You are helping a local AI assistant search the web.\n"
        f"Today's date is {today}. Your training data is outdated, so always "
        "use this date to pick the correct year in queries about current "
        "events, seasons, prices, or versions. Never use a year from memory.\n"
        "Generate 3 to 5 concise search engine queries that maximize retrieval quality.\n"
        "Infer the domain from the question.\n"
        "Prefer keyword-style search queries over full natural-language questions.\n"
        "Remove conversational filler words and question words.\n"
        "For non-English questions, translate the search intent to English unless the user explicitly needs sources in that language.\n"
        "For current-status questions, include terms such as latest, update, news, status, or report when useful.\n"
        "For people, companies, teams, products, or events, preserve exact named entities and add relevant domain terms when inferable.\n"
        "For technical questions, include specific domain terminology, likely keywords from documentation/articles, and avoid generic wording.\n"
        "Prefer web-friendly wording and keep named entities exact.\n"
        "If the user's question is in Spanish, you may output queries in English when that improves web search quality.\n"
        "Return only the queries, one per line, with no numbering, no bullets, and no explanation.\n\n"
        "Examples:\n"
        "User question: how can I optimize a SQL query\n"
        "sql query optimization best practices\n"
        "how to optimize sql queries performance\n"
        "sql query tuning indexing explain analyze\n"
        "postgres query optimization guide\n\n"
        "User question: is Snowflake losing customers?\n"
        f"Snowflake customer churn {date.today().year}\n"
        f"Snowflake customer count quarterly results {date.today().year}\n"
        f"Snowflake revenue growth customers {date.today().year}\n"
        f"analyst report Snowflake demand trends {date.today().year}\n\n"
        f"User question: {question}\n"
    )
    raw_output = model(planner_prompt)
    parsed_queries = parse_query_lines(raw_output)
    return finalize_queries(parsed_queries, fallback_query=question, limit=limit)
