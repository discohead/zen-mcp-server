"""Tests for the GitHub Copilot CLI agent."""

import asyncio
import shutil
from pathlib import Path

import pytest

from clink.agents.base import CLIAgentError
from clink.agents.copilot import CopilotAgent
from clink.models import ResolvedCLIClient, ResolvedCLIRole


class DummyProcess:
    """Mock subprocess for testing."""

    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.communicated = False

    async def communicate(self):
        """Copilot agent doesn't pass stdin data - it uses -p flag."""
        self.communicated = True
        return self._stdout, self._stderr


@pytest.fixture()
def copilot_agent():
    """Create a CopilotAgent with test configuration."""
    prompt_path = Path("systemprompts/clink/default.txt").resolve()
    role = ResolvedCLIRole(name="default", prompt_path=prompt_path, role_args=[])
    client = ResolvedCLIClient(
        name="copilot",
        executable=["copilot"],
        internal_args=[],
        config_args=["--allow-all-tools"],
        env={},
        timeout_seconds=30,
        parser="copilot_text",
        runner="copilot",
        roles={"default": role},
        output_to_file=None,
        working_dir=None,
    )
    return CopilotAgent(client), role


async def _run_agent_with_process(monkeypatch, agent, role, process, *, prompt="Test prompt", system_prompt=None):
    """Helper to run agent with mocked subprocess."""
    captured_args = {}

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured_args["args"] = args
        captured_args["kwargs"] = kwargs
        return process

    def fake_which(executable_name):
        return f"/usr/bin/{executable_name}"

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(shutil, "which", fake_which)

    result = await agent.run(
        role=role,
        prompt=prompt,
        system_prompt=system_prompt,
        files=[],
        images=[],
    )
    return result, captured_args


@pytest.mark.asyncio
async def test_copilot_agent_passes_prompt_via_p_flag(monkeypatch, copilot_agent):
    """Test that prompt is passed via -p flag, not stdin."""
    agent, role = copilot_agent
    process = DummyProcess(stdout=b"Response from Copilot")

    result, captured_args = await _run_agent_with_process(
        monkeypatch, agent, role, process, prompt="List all files"
    )

    # Verify -p flag is in the command
    command_args = captured_args["args"]
    assert "-p" in command_args
    p_idx = command_args.index("-p")
    assert command_args[p_idx + 1] == "List all files"

    # Verify stdin is DEVNULL (no stdin data passed)
    assert captured_args["kwargs"]["stdin"] == asyncio.subprocess.DEVNULL


@pytest.mark.asyncio
async def test_copilot_agent_prepends_system_prompt(monkeypatch, copilot_agent):
    """Test that system prompt is prepended to user prompt."""
    agent, role = copilot_agent
    process = DummyProcess(stdout=b"Response")

    result, captured_args = await _run_agent_with_process(
        monkeypatch,
        agent,
        role,
        process,
        prompt="User request",
        system_prompt="Be concise and helpful.",
    )

    # Find the prompt in the command args
    command_args = captured_args["args"]
    p_idx = command_args.index("-p")
    full_prompt = command_args[p_idx + 1]

    # System prompt should be prepended
    assert full_prompt.startswith("Be concise and helpful.")
    assert "User request" in full_prompt
    assert full_prompt == "Be concise and helpful.\n\nUser request"


@pytest.mark.asyncio
async def test_copilot_agent_includes_config_args(monkeypatch, copilot_agent):
    """Test that config args are included in command."""
    agent, role = copilot_agent
    process = DummyProcess(stdout=b"Response")

    result, captured_args = await _run_agent_with_process(monkeypatch, agent, role, process)

    command_args = captured_args["args"]
    assert "--allow-all-tools" in command_args


@pytest.mark.asyncio
async def test_copilot_agent_parses_text_output(monkeypatch, copilot_agent):
    """Test that plain text output is parsed correctly."""
    agent, role = copilot_agent
    process = DummyProcess(stdout=b"Here is your code:\n```python\nprint('hello')\n```")

    result, _ = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.returncode == 0
    assert "Here is your code:" in result.parsed.content
    assert "print('hello')" in result.parsed.content
    assert result.parser_name == "copilot_text"


@pytest.mark.asyncio
async def test_copilot_agent_recovers_from_error_with_output(monkeypatch, copilot_agent):
    """Test that agent recovers useful output even on non-zero exit code."""
    agent, role = copilot_agent
    process = DummyProcess(
        stdout=b"Partial response before timeout occurred",
        returncode=124,  # Timeout exit code
    )

    result, _ = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.returncode == 124
    assert "Partial response" in result.parsed.content
    assert result.parsed.metadata.get("cli_exit_code") == 124
    assert result.parsed.metadata.get("cli_error_recovered") is True


@pytest.mark.asyncio
async def test_copilot_agent_propagates_empty_error(monkeypatch, copilot_agent):
    """Test that empty output on error raises CLIAgentError."""
    agent, role = copilot_agent
    process = DummyProcess(stdout=b"", stderr=b"Fatal error", returncode=1)

    with pytest.raises(CLIAgentError) as exc_info:
        await _run_agent_with_process(monkeypatch, agent, role, process)

    assert "exited with status 1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_copilot_agent_sanitizes_long_prompts(monkeypatch, copilot_agent):
    """Test that long prompts are truncated in sanitized command."""
    agent, role = copilot_agent
    long_prompt = "x" * 500  # 500 chars, exceeds 200 char limit for display
    process = DummyProcess(stdout=b"Response")

    result, _ = await _run_agent_with_process(
        monkeypatch, agent, role, process, prompt=long_prompt
    )

    # The sanitized command should have truncated prompt
    p_idx = result.sanitized_command.index("-p")
    sanitized_prompt = result.sanitized_command[p_idx + 1]
    assert len(sanitized_prompt) < 500
    assert sanitized_prompt.endswith("...[truncated]")


@pytest.mark.asyncio
async def test_copilot_agent_handles_stderr_warnings(monkeypatch, copilot_agent):
    """Test that stderr warnings are captured in metadata."""
    agent, role = copilot_agent
    process = DummyProcess(
        stdout=b"Successful response",
        stderr=b"Warning: deprecated feature used",
        returncode=0,
    )

    result, _ = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.parsed.content == "Successful response"
    assert result.stderr == "Warning: deprecated feature used"


@pytest.mark.asyncio
async def test_copilot_agent_no_system_prompt(monkeypatch, copilot_agent):
    """Test that agent works without system prompt."""
    agent, role = copilot_agent
    process = DummyProcess(stdout=b"Response")

    result, captured_args = await _run_agent_with_process(
        monkeypatch, agent, role, process, prompt="Just a prompt", system_prompt=None
    )

    command_args = captured_args["args"]
    p_idx = command_args.index("-p")
    full_prompt = command_args[p_idx + 1]

    # Without system prompt, just the user prompt
    assert full_prompt == "Just a prompt"


@pytest.mark.asyncio
async def test_copilot_agent_empty_system_prompt(monkeypatch, copilot_agent):
    """Test that empty system prompt is handled."""
    agent, role = copilot_agent
    process = DummyProcess(stdout=b"Response")

    result, captured_args = await _run_agent_with_process(
        monkeypatch, agent, role, process, prompt="User prompt", system_prompt="   "
    )

    command_args = captured_args["args"]
    p_idx = command_args.index("-p")
    full_prompt = command_args[p_idx + 1]

    # Empty/whitespace system prompt should be ignored
    assert full_prompt == "User prompt"


@pytest.mark.asyncio
async def test_copilot_agent_executable_not_found(monkeypatch, copilot_agent):
    """Test that missing executable raises CLIAgentError with helpful message."""
    agent, role = copilot_agent

    def fake_which(_):
        return None

    monkeypatch.setattr(shutil, "which", fake_which)

    with pytest.raises(CLIAgentError) as exc_info:
        await agent.run(role=role, prompt="Test", system_prompt=None, files=[], images=[])

    assert "not found in PATH" in str(exc_info.value)
    assert "npm install -g @github/copilot" in str(exc_info.value)


@pytest.mark.asyncio
async def test_copilot_agent_timeout(monkeypatch, copilot_agent):
    """Test that timeout is properly handled."""
    agent, role = copilot_agent

    async def slow_communicate():
        await asyncio.sleep(10)  # Simulate slow response
        return b"", b""

    class SlowProcess:
        returncode = None

        async def communicate(self):
            return await slow_communicate()

        def kill(self):
            pass

    async def fake_create_subprocess_exec(*args, **kwargs):
        return SlowProcess()

    def fake_which(executable_name):
        return f"/usr/bin/{executable_name}"

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(shutil, "which", fake_which)

    # Set a very short timeout for testing (1 second minimum, as it must be int)
    agent.client = ResolvedCLIClient(
        name="copilot",
        executable=["copilot"],
        internal_args=[],
        config_args=["--allow-all-tools"],
        env={},
        timeout_seconds=1,  # 1 second timeout (minimum int value)
        parser="copilot_text",
        runner="copilot",
        roles={"default": role},
        output_to_file=None,
        working_dir=None,
    )

    with pytest.raises(CLIAgentError) as exc_info:
        await agent.run(role=role, prompt="Test", system_prompt=None, files=[], images=[])

    assert "timed out" in str(exc_info.value)
