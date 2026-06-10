"""The prompts must carry today's date so the model stops living in 2023."""

from datetime import date

from llm.qwen_runner import build_direct_answer_prompt
from llm.web_rag import WebRagPipeline


def test_direct_answer_prompt_includes_today():
    prompt = build_direct_answer_prompt("any question")
    assert date.today().isoformat() in prompt


def test_web_rag_prompt_includes_today():
    prompt = WebRagPipeline()._build_answer_prompt("any question", "context")
    assert date.today().isoformat() in prompt


def test_planner_prompt_includes_today():
    captured: list[str] = []

    def fake_model(prompt: str) -> str:
        captured.append(prompt)
        return "some query"

    from llm.query_planner import generate_search_queries

    generate_search_queries("any question", model=fake_model)
    assert date.today().isoformat() in captured[0]
    assert "2023" not in captured[0]
