"""Decide whether a question needs web search or a local answer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from llm.qwen_runner import ask_model

WEB = "WEB"
LOCAL = "LOCAL"

ROUTER_PROMPT_TEMPLATE = (
    "You are a decision router for a local AI assistant.\n"
    "Your job is to decide whether the assistant needs web search.\n\n"
    "Return only valid JSON:\n"
    '{"decision": "WEB" or "LOCAL", "reason": "short reason"}\n\n'
    "Choose WEB when the answer depends on:\n"
    "- current or recent facts\n"
    "- sports, injuries, standings, schedules\n"
    "- news\n"
    "- prices\n"
    "- laws or regulations\n"
    "- software versions or releases\n"
    "- company/person status\n"
    "- anything happening today, now, this week, this season, recently, currently\n\n"
    "Choose LOCAL when the answer is about:\n"
    "- explaining code\n"
    "- programming concepts\n"
    "- math\n"
    "- general knowledge\n"
    "- writing help\n"
    "- translations\n"
    "- stable concepts\n\n"
    "User question: {question}\n"
)


@dataclass
class RouterDecision:
    """Result of routing one question."""

    decision: str  # WEB or LOCAL
    reason: str


class AssistantRouter:
    """Use the local model to choose between web search and a local answer."""

    def decide(self, question: str) -> RouterDecision:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        prompt = ROUTER_PROMPT_TEMPLATE.replace("{question}", question)

        try:
            raw_output = ask_model(prompt)
        except Exception:
            # If the model fails, searching too much is better than inventing.
            return RouterDecision(
                decision=WEB,
                reason="Router model failed, defaulting to web search.",
            )

        return parse_router_output(raw_output)

    def should_search_web(self, question: str) -> bool:
        return self.decide(question).decision == WEB


def parse_router_output(raw_output: str) -> RouterDecision:
    """Read the model output and extract the decision.

    Small local models often wrap JSON in extra text or code fences,
    so we look for the first JSON object instead of parsing directly.
    Anything we cannot understand defaults to WEB.
    """
    json_match = re.search(r"\{.*?\}", raw_output, re.DOTALL)
    if not json_match:
        return _fallback_decision(raw_output)

    try:
        parsed = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return _fallback_decision(raw_output)

    decision = str(parsed.get("decision", "")).strip().upper()
    reason = str(parsed.get("reason", "")).strip() or "No reason given."

    if decision not in {WEB, LOCAL}:
        return _fallback_decision(raw_output)

    return RouterDecision(decision=decision, reason=reason)


def _fallback_decision(raw_output: str) -> RouterDecision:
    # Last attempt: the model may have answered with a plain word.
    normalized_output = raw_output.strip().upper()
    if LOCAL in normalized_output and WEB not in normalized_output:
        return RouterDecision(
            decision=LOCAL,
            reason="Model answered LOCAL without valid JSON.",
        )

    return RouterDecision(
        decision=WEB,
        reason="Could not parse router output, defaulting to web search.",
    )
