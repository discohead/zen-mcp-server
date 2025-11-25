"""Tests for the GitHub Copilot CLI text parser."""

import pytest

from clink.parsers.base import ParserError
from clink.parsers.copilot import CopilotTextParser


def test_copilot_parser_extracts_plain_text():
    """Test that the parser correctly extracts plain text content."""
    parser = CopilotTextParser()
    stdout = "This is the response from Copilot CLI.\nIt spans multiple lines."

    parsed = parser.parse(stdout=stdout, stderr="")

    assert parsed.content == "This is the response from Copilot CLI.\nIt spans multiple lines."
    assert parsed.metadata == {}


def test_copilot_parser_strips_whitespace():
    """Test that the parser strips leading/trailing whitespace."""
    parser = CopilotTextParser()
    stdout = "   \n\n  Response with whitespace  \n\n   "

    parsed = parser.parse(stdout=stdout, stderr="")

    assert parsed.content == "Response with whitespace"


def test_copilot_parser_includes_stderr_in_metadata():
    """Test that stderr is captured in metadata when present."""
    parser = CopilotTextParser()
    stdout = "Normal response"
    stderr = "Warning: some deprecation notice"

    parsed = parser.parse(stdout=stdout, stderr=stderr)

    assert parsed.content == "Normal response"
    assert parsed.metadata["stderr"] == "Warning: some deprecation notice"


def test_copilot_parser_requires_output():
    """Test that empty stdout raises ParserError."""
    parser = CopilotTextParser()

    with pytest.raises(ParserError) as exc_info:
        parser.parse(stdout="", stderr="")

    assert "empty output" in str(exc_info.value).lower()


def test_copilot_parser_requires_output_whitespace_only():
    """Test that whitespace-only stdout raises ParserError."""
    parser = CopilotTextParser()

    with pytest.raises(ParserError):
        parser.parse(stdout="   \n\t  ", stderr="")


def test_copilot_parser_handles_empty_stdout_with_stderr():
    """Test that empty stdout with stderr content returns message about it."""
    parser = CopilotTextParser()
    stderr = "Error: authentication failed"

    parsed = parser.parse(stdout="", stderr=stderr)

    assert "no stdout" in parsed.content.lower()
    assert parsed.metadata["stderr"] == "Error: authentication failed"
    assert parsed.metadata["empty_stdout"] is True


def test_copilot_parser_detects_error_keyword():
    """Test that error keywords in content set possible_error metadata."""
    parser = CopilotTextParser()
    stdout = "Error: Unable to connect to GitHub"

    parsed = parser.parse(stdout=stdout, stderr="")

    assert parsed.content == "Error: Unable to connect to GitHub"
    assert parsed.metadata.get("possible_error") is True


def test_copilot_parser_detects_error_dash_format():
    """Test that 'error -' format in content sets possible_error metadata."""
    parser = CopilotTextParser()
    stdout = "Something went wrong\nerror - network timeout occurred"

    parsed = parser.parse(stdout=stdout, stderr="")

    assert parsed.metadata.get("possible_error") is True


def test_copilot_parser_detects_auth_issues():
    """Test that authentication keywords set auth_issue metadata."""
    parser = CopilotTextParser()
    stdout = "Please authenticate with GitHub to continue"

    parsed = parser.parse(stdout=stdout, stderr="")

    assert parsed.metadata.get("auth_issue") is True


def test_copilot_parser_detects_login_keyword():
    """Test that login keyword sets auth_issue metadata."""
    parser = CopilotTextParser()
    stdout = "You need to login first using /login command"

    parsed = parser.parse(stdout=stdout, stderr="")

    assert parsed.metadata.get("auth_issue") is True


def test_copilot_parser_detects_permission_denied():
    """Test that permission denied content sets permission_denied metadata."""
    parser = CopilotTextParser()
    stdout = "Permission denied to access this repository"

    parsed = parser.parse(stdout=stdout, stderr="")

    assert parsed.metadata.get("permission_denied") is True


def test_copilot_parser_normal_content_no_flags():
    """Test that normal content doesn't set error/auth flags."""
    parser = CopilotTextParser()
    stdout = "Here is your code review:\n- Good structure\n- Clean code"

    parsed = parser.parse(stdout=stdout, stderr="")

    assert parsed.content == "Here is your code review:\n- Good structure\n- Clean code"
    assert "possible_error" not in parsed.metadata
    assert "auth_issue" not in parsed.metadata
    assert "permission_denied" not in parsed.metadata


def test_copilot_parser_case_insensitive_detection():
    """Test that error detection is case insensitive."""
    parser = CopilotTextParser()
    stdout = "ERROR: Something failed"

    parsed = parser.parse(stdout=stdout, stderr="")

    assert parsed.metadata.get("possible_error") is True


def test_copilot_parser_preserves_multiline_content():
    """Test that multiline content is preserved correctly."""
    parser = CopilotTextParser()
    stdout = """# Code Review

## Summary
The code looks good overall.

## Issues Found
1. Missing null check on line 42
2. Unused import on line 5

## Recommendations
- Add error handling
- Consider using TypeScript"""

    parsed = parser.parse(stdout=stdout, stderr="")

    assert "# Code Review" in parsed.content
    assert "Missing null check" in parsed.content
    assert "Consider using TypeScript" in parsed.content
