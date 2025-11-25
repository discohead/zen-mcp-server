"""GitHub Copilot CLI-specific agent hooks."""

from __future__ import annotations

import asyncio
import shutil
import time
from collections.abc import Sequence

from clink.constants import DEFAULT_STREAM_LIMIT
from clink.models import ResolvedCLIRole
from clink.parsers.base import ParserError

from .base import AgentOutput, BaseCLIAgent, CLIAgentError


class CopilotAgent(BaseCLIAgent):
    """GitHub Copilot CLI agent with prompt-via-argument support.

    Unlike other CLI agents that receive prompts via stdin, the GitHub Copilot
    CLI uses the `-p` or `--prompt` flag to receive prompts as command-line
    arguments. This agent overrides the run method to handle this difference.
    """

    async def run(
        self,
        *,
        role: ResolvedCLIRole,
        prompt: str,
        system_prompt: str | None = None,
        files: Sequence[str],
        images: Sequence[str],
    ) -> AgentOutput:
        """Execute Copilot CLI with prompt passed via -p argument."""
        # Files and images are already embedded into the prompt by the tool
        _ = (files, images)

        # Build the command with the prompt included as an argument
        command = self._build_command_with_prompt(
            role=role,
            system_prompt=system_prompt,
            prompt=prompt,
        )
        env = self._build_environment()

        # Resolve executable path for cross-platform compatibility
        executable_name = command[0]
        resolved_executable = shutil.which(executable_name)
        if resolved_executable is None:
            raise CLIAgentError(
                f"Executable '{executable_name}' not found in PATH for CLI '{self.client.name}'. "
                f"Ensure GitHub Copilot CLI is installed via 'npm install -g @github/copilot'."
            )
        command[0] = resolved_executable

        sanitized_command = self._sanitize_command(command, prompt)

        cwd = str(self.client.working_dir) if self.client.working_dir else None
        limit = DEFAULT_STREAM_LIMIT

        stdout_text = ""
        stderr_text = ""
        start_time = time.monotonic()

        self._logger.debug("Executing Copilot CLI command: %s", " ".join(sanitized_command))
        if cwd:
            self._logger.debug("Working directory: %s", cwd)

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,  # No stdin needed - prompt via -p
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                limit=limit,
                env=env,
            )
        except FileNotFoundError as exc:
            raise CLIAgentError(f"Executable not found for CLI '{self.client.name}': {exc}") from exc

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self.client.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise CLIAgentError(
                f"CLI '{self.client.name}' timed out after {self.client.timeout_seconds} seconds",
                returncode=None,
            ) from exc

        duration = time.monotonic() - start_time
        return_code = process.returncode
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")

        if return_code != 0:
            recovered = self._recover_from_error(
                returncode=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
                sanitized_command=sanitized_command,
                duration_seconds=duration,
                output_file_content=None,
            )
            if recovered is not None:
                return recovered

        if return_code != 0:
            raise CLIAgentError(
                f"CLI '{self.client.name}' exited with status {return_code}",
                returncode=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
            )

        try:
            parsed = self._parser.parse(stdout_text, stderr_text)
        except ParserError as exc:
            raise CLIAgentError(
                f"Failed to parse output from CLI '{self.client.name}': {exc}",
                returncode=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
            ) from exc

        return AgentOutput(
            parsed=parsed,
            sanitized_command=sanitized_command,
            returncode=return_code,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_seconds=duration,
            parser_name=self._parser.name,
            output_file_content=None,
        )

    def _build_command_with_prompt(
        self,
        *,
        role: ResolvedCLIRole,
        system_prompt: str | None,
        prompt: str,
    ) -> list[str]:
        """Build command with prompt as -p argument."""
        command = list(self.client.executable)
        command.extend(self.client.internal_args)
        command.extend(self.client.config_args)
        command.extend(role.role_args)

        # Add the prompt via -p flag (Copilot CLI's programmatic mode)
        # The system prompt is prepended to the user prompt since Copilot
        # doesn't have a separate system prompt mechanism
        full_prompt = prompt
        if system_prompt and system_prompt.strip():
            full_prompt = f"{system_prompt.strip()}\n\n{prompt}"

        command.extend(["-p", full_prompt])

        return command

    def _sanitize_command(self, command: list[str], prompt: str) -> list[str]:
        """Create a sanitized version of the command for logging/metadata.

        Truncates the prompt to avoid excessive metadata size.
        """
        sanitized = list(command)
        try:
            prompt_idx = sanitized.index("-p")
            if prompt_idx + 1 < len(sanitized):
                # Truncate the prompt for display purposes
                displayed_prompt = sanitized[prompt_idx + 1]
                if len(displayed_prompt) > 200:
                    sanitized[prompt_idx + 1] = displayed_prompt[:200] + "...[truncated]"
        except ValueError:
            pass
        return sanitized

    def _recover_from_error(
        self,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        sanitized_command: list[str],
        duration_seconds: float,
        output_file_content: str | None,
    ) -> AgentOutput | None:
        """Attempt to recover useful output from failed executions.

        Copilot CLI may exit with non-zero status but still provide useful
        output (e.g., partial responses, warnings, or informative error messages).
        """
        # If there's stdout content, try to parse it even on error
        if stdout and stdout.strip():
            try:
                parsed = self._parser.parse(stdout, stderr)
                # Mark the response as having an error
                parsed.metadata["cli_exit_code"] = returncode
                parsed.metadata["cli_error_recovered"] = True
                return AgentOutput(
                    parsed=parsed,
                    sanitized_command=sanitized_command,
                    returncode=returncode,
                    stdout=stdout,
                    stderr=stderr,
                    duration_seconds=duration_seconds,
                    parser_name=self._parser.name,
                    output_file_content=output_file_content,
                )
            except ParserError:
                pass

        return None
