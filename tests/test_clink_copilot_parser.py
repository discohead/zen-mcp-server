"""Tests for the Copilot CLI JSON parser."""

import json

import pytest

from clink.parsers.base import ParserError
from clink.parsers.copilot import CopilotJSONParser


def test_copilot_parser_extracts_response_content():
    parser = CopilotJSONParser()
    stdout = json.dumps(
        {
            "response": "Hello from Copilot",
            "session_id": "session123",
            "cost_stats": {"tokens_used": 42},
        }
    )

    parsed = parser.parse(stdout=stdout, stderr="")

    assert parsed.content == "Hello from Copilot"
    assert parsed.metadata["session_id"] == "session123"
    assert parsed.metadata["cost_stats"]["tokens_used"] == 42
    assert parsed.metadata["tokens_used"] == 42


def test_copilot_parser_extracts_result_content():
    parser = CopilotJSONParser()
    stdout = json.dumps({"result": "Result from Copilot"})

    parsed = parser.parse(stdout=stdout, stderr="")

    assert parsed.content == "Result from Copilot"


def test_copilot_parser_extracts_from_messages():
    parser = CopilotJSONParser()
    stdout = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "assistant", "content": "How can I help?"},
            ]
        }
    )

    parsed = parser.parse(stdout=stdout, stderr="")

    assert "Hi there!" in parsed.content
    assert "How can I help?" in parsed.content


def test_copilot_parser_extracts_from_tasks():
    parser = CopilotJSONParser()
    stdout = json.dumps(
        {
            "tasks": [
                {"type": "tool_call", "tool": "bash", "result": "file1.py"},
                {"type": "tool_call", "tool": "bash", "result": "file2.py"},
            ]
        }
    )

    parsed = parser.parse(stdout=stdout, stderr="")

    assert "file1.py" in parsed.content
    assert "file2.py" in parsed.content


def test_copilot_parser_extracts_message_content():
    parser = CopilotJSONParser()
    stdout = json.dumps({"message": "A direct message"})

    parsed = parser.parse(stdout=stdout, stderr="")

    assert parsed.content == "A direct message"


def test_copilot_parser_extracts_direct_content():
    parser = CopilotJSONParser()
    stdout = json.dumps({"content": "Direct content field"})

    parsed = parser.parse(stdout=stdout, stderr="")

    assert parsed.content == "Direct content field"


def test_copilot_parser_requires_output():
    parser = CopilotJSONParser()

    with pytest.raises(ParserError):
        parser.parse(stdout="", stderr="")


def test_copilot_parser_handles_invalid_json():
    parser = CopilotJSONParser()

    with pytest.raises(ParserError) as exc_info:
        parser.parse(stdout="not valid json", stderr="")

    assert "Failed to decode Copilot CLI JSON output" in str(exc_info.value)


def test_copilot_parser_handles_empty_response_with_stderr():
    parser = CopilotJSONParser()
    stdout = json.dumps({})

    parsed = parser.parse(stdout=stdout, stderr="Some warning")

    assert "Raw stderr was preserved" in parsed.content
    assert parsed.metadata["stderr"] == "Some warning"


def test_copilot_parser_raises_when_no_content():
    parser = CopilotJSONParser()
    stdout = json.dumps({"unrelated": "data"})

    with pytest.raises(ParserError) as exc_info:
        parser.parse(stdout=stdout, stderr="")

    assert "did not contain a textual result" in str(exc_info.value)


def test_copilot_parser_preserves_stderr():
    parser = CopilotJSONParser()
    stdout = json.dumps({"response": "Hello"})

    parsed = parser.parse(stdout=stdout, stderr="Warning message")

    assert parsed.content == "Hello"
    assert parsed.metadata["stderr"] == "Warning message"
