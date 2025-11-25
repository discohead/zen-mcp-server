"""Parser for Copilot CLI JSON output."""

from __future__ import annotations

import json
from typing import Any

from .base import BaseParser, ParsedCLIResponse, ParserError


class CopilotJSONParser(BaseParser):
    """Parse stdout produced by `copilot --print --json`."""

    name = "copilot_json"

    def parse(self, stdout: str, stderr: str) -> ParsedCLIResponse:
        if not stdout.strip():
            raise ParserError("Copilot CLI returned empty stdout while JSON output was expected")

        try:
            payload: dict[str, Any] = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise ParserError(f"Failed to decode Copilot CLI JSON output: {exc}") from exc

        metadata: dict[str, Any] = {"raw": payload}

        # Extract content from various possible locations in the response
        content = self._extract_content(payload)

        # Extract session ID if present
        session_id = payload.get("session_id")
        if isinstance(session_id, str) and session_id:
            metadata["session_id"] = session_id

        # Extract usage/cost statistics if present
        cost_stats = payload.get("cost_stats")
        if isinstance(cost_stats, dict):
            metadata["cost_stats"] = cost_stats
            tokens_used = cost_stats.get("tokens_used")
            if isinstance(tokens_used, int):
                metadata["tokens_used"] = tokens_used

        # Extract messages if present
        messages = payload.get("messages")
        if isinstance(messages, list):
            metadata["messages"] = messages

        # Extract tasks if present
        tasks = payload.get("tasks")
        if isinstance(tasks, list):
            metadata["tasks"] = tasks

        if stderr and stderr.strip():
            metadata["stderr"] = stderr.strip()

        if content:
            return ParsedCLIResponse(content=content, metadata=metadata)

        # If no content found but we have stderr, use it for troubleshooting
        if stderr and stderr.strip():
            return ParsedCLIResponse(
                content="Copilot CLI returned no textual result. Raw stderr was preserved for troubleshooting.",
                metadata=metadata,
            )

        raise ParserError("Copilot CLI response did not contain a textual result")

    def _extract_content(self, payload: dict[str, Any]) -> str:
        """Extract the main content from the Copilot CLI response."""
        # Try direct response field first
        response = payload.get("response")
        if isinstance(response, str) and response.strip():
            return response.strip()

        # Try result field
        result = payload.get("result")
        if isinstance(result, str) and result.strip():
            return result.strip()

        # Try extracting from messages array (look for assistant messages)
        messages = payload.get("messages")
        if isinstance(messages, list):
            assistant_contents: list[str] = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role")
                    content = msg.get("content")
                    if role == "assistant" and isinstance(content, str) and content.strip():
                        assistant_contents.append(content.strip())
            if assistant_contents:
                return "\n\n".join(assistant_contents)

        # Try extracting from tasks array (look for tool results)
        tasks = payload.get("tasks")
        if isinstance(tasks, list):
            results: list[str] = []
            for task in tasks:
                if isinstance(task, dict):
                    task_result = task.get("result")
                    if isinstance(task_result, str) and task_result.strip():
                        results.append(task_result.strip())
            if results:
                return "\n\n".join(results)

        # Try message field
        message = payload.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()

        # Try content field directly
        content = payload.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

        return ""
