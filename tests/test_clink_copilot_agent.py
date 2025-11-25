"""Tests for the Copilot CLI agent."""

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from clink.agents.base import CLIAgentError
from clink.agents.copilot import CopilotAgent
from clink.models import ResolvedCLIClient, ResolvedCLIRole


class DummyProcess:
    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.stdin_data: bytes | None = None

    async def communicate(self, input_data):
        self.stdin_data = input_data
        return self._stdout, self._stderr


@pytest.fixture()
def copilot_agent():
    prompt_path = Path("systemprompts/clink/default.txt").resolve()
    role = ResolvedCLIRole(name="default", prompt_path=prompt_path, role_args=[])
    client = ResolvedCLIClient(
        name="copilot",
        executable=["copilot"],
        internal_args=["--print", "--json"],
        config_args=[],
        env={},
        timeout_seconds=30,
        parser="copilot_json",
        runner="copilot",
        roles={"default": role},
        output_to_file=None,
        working_dir=None,
    )
    return CopilotAgent(client), role


async def _run_agent_with_process(monkeypatch, agent, role, process):
    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return process

    def fake_which(executable_name):
        return f"/usr/bin/{executable_name}"

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(shutil, "which", fake_which)

    return await agent.run(
        role=role,
        prompt="Hello Copilot",
        system_prompt=None,
        files=[],
        images=[],
    )


@pytest.mark.asyncio
async def test_copilot_agent_parses_success_response(monkeypatch, copilot_agent):
    agent, role = copilot_agent
    stdout_payload = json.dumps(
        {
            "response": "Hello from Copilot!",
            "session_id": "session123",
            "cost_stats": {"tokens_used": 100},
        }
    ).encode()
    process = DummyProcess(stdout=stdout_payload)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.returncode == 0
    assert result.parsed.content == "Hello from Copilot!"
    assert result.parsed.metadata["session_id"] == "session123"
    assert result.parsed.metadata["tokens_used"] == 100


@pytest.mark.asyncio
async def test_copilot_agent_recovers_from_error_with_valid_output(monkeypatch, copilot_agent):
    agent, role = copilot_agent
    stdout_payload = json.dumps(
        {
            "response": "Partial response before error",
            "session_id": "session456",
        }
    ).encode()
    process = DummyProcess(stdout=stdout_payload, returncode=1)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.returncode == 1
    assert result.parsed.content == "Partial response before error"
    assert result.parsed.metadata["session_id"] == "session456"


@pytest.mark.asyncio
async def test_copilot_agent_propagates_unparseable_output(monkeypatch, copilot_agent):
    agent, role = copilot_agent
    process = DummyProcess(stdout=b"", returncode=1)

    with pytest.raises(CLIAgentError):
        await _run_agent_with_process(monkeypatch, agent, role, process)


@pytest.mark.asyncio
async def test_copilot_agent_handles_messages_response(monkeypatch, copilot_agent):
    agent, role = copilot_agent
    stdout_payload = json.dumps(
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"},
            ]
        }
    ).encode()
    process = DummyProcess(stdout=stdout_payload)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.returncode == 0
    assert "Hi! How can I help?" in result.parsed.content
