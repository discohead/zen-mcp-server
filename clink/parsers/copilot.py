"""Parser for GitHub Copilot CLI text output."""

from __future__ import annotations

from typing import Any

from .base import BaseParser, ParsedCLIResponse, ParserError


class CopilotTextParser(BaseParser):
    """Parse stdout produced by `copilot -p`.

    GitHub Copilot CLI outputs plain text (no JSON mode available).
    This parser captures the output as-is and extracts any available
    metadata from the response structure.
    """

    name = "copilot_text"

    def parse(self, stdout: str, stderr: str) -> ParsedCLIResponse:
        # Copilot CLI outputs plain text, not JSON
        content = (stdout or "").strip()

        if not content:
            # Check stderr for error messages
            stderr_text = (stderr or "").strip()
            if stderr_text:
                # If there's stderr output but no stdout, treat it as a message
                return ParsedCLIResponse(
                    content=f"Copilot CLI returned no stdout. Stderr: {stderr_text}",
                    metadata={"stderr": stderr_text, "empty_stdout": True},
                )
            raise ParserError("Copilot CLI returned empty output")

        metadata: dict[str, Any] = {}

        # Extract any stderr warnings
        stderr_text = (stderr or "").strip()
        if stderr_text:
            metadata["stderr"] = stderr_text

        # Try to detect if the response indicates an error condition
        lower_content = content.lower()
        if "error:" in lower_content or "error -" in lower_content:
            metadata["possible_error"] = True

        # Check for authentication issues
        if "authenticate" in lower_content or "login" in lower_content:
            metadata["auth_issue"] = True

        # Check for permission issues
        if "permission" in lower_content and "denied" in lower_content:
            metadata["permission_denied"] = True

        return ParsedCLIResponse(content=content, metadata=metadata)
