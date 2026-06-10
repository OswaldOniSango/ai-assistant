"""Tests for the WEB/LOCAL decision router."""

from llm.assistant_router import LOCAL, WEB, parse_router_output


def test_parses_clean_json_web():
    output = '{"decision": "WEB", "reason": "asks about current standings"}'
    result = parse_router_output(output)
    assert result.decision == WEB
    assert result.reason == "asks about current standings"


def test_parses_clean_json_local():
    output = '{"decision": "LOCAL", "reason": "programming concept"}'
    result = parse_router_output(output)
    assert result.decision == LOCAL


def test_parses_json_wrapped_in_code_fences():
    output = '```json\n{"decision": "LOCAL", "reason": "math question"}\n```'
    result = parse_router_output(output)
    assert result.decision == LOCAL


def test_parses_json_with_extra_text_around_it():
    output = 'Sure! Here is my answer:\n{"decision": "WEB", "reason": "news"}\nHope it helps.'
    result = parse_router_output(output)
    assert result.decision == WEB


def test_lowercase_decision_is_normalized():
    output = '{"decision": "local", "reason": "stable concept"}'
    result = parse_router_output(output)
    assert result.decision == LOCAL


def test_plain_word_local_without_json():
    result = parse_router_output("LOCAL")
    assert result.decision == LOCAL


def test_unparseable_output_defaults_to_web():
    result = parse_router_output("I am not sure what to answer here.")
    assert result.decision == WEB


def test_invalid_decision_value_defaults_to_web():
    output = '{"decision": "MAYBE", "reason": "unsure"}'
    result = parse_router_output(output)
    assert result.decision == WEB


def test_broken_json_defaults_to_web():
    output = '{"decision": "WEB", "reason": '
    result = parse_router_output(output)
    assert result.decision == WEB
