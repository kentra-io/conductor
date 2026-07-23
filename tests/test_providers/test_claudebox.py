"""Hermetic tests for ClaudeboxProvider (M1) against a fake `cb` executable.

No real claudebox box, no real `claude` CLI, no network, no LLM. The fake
`cb` script (materialized per-test into ``tmp_path``) stands in for the
real `cb` binary and emits canned ``stream-json`` lines controlled by the
``FAKE_CB_MODE`` environment variable, so every branch of
``ClaudeboxProvider.execute()`` can be exercised deterministically.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from conductor.config.schema import AgentDef, OutputField
from conductor.exceptions import ProviderError, ValidationError
from conductor.providers.claudebox import ClaudeboxProvider

pytestmark = pytest.mark.asyncio

_FAKE_CB_SCRIPT = f"""\
#!{sys.executable}
import json
import os
import sys
import time


def emit(obj):
    print(json.dumps(obj), flush=True)


def main():
    argv = sys.argv[1:]

    argv_dump = os.environ.get("FAKE_CB_ARGV_DUMP")
    if argv_dump:
        with open(argv_dump, "a", encoding="utf-8") as f:
            f.write(json.dumps(argv) + chr(10))

    if argv[:1] == ["ls"]:
        sys.exit(int(os.environ.get("FAKE_CB_LS_EXIT_CODE", "0")))

    mode = os.environ.get("FAKE_CB_MODE", "normal")
    session_id = os.environ.get("FAKE_CB_SESSION_ID", "sess-fake-1")

    if mode == "sleep":
        time.sleep(float(os.environ.get("FAKE_CB_SLEEP_SECONDS", "30")))
        sys.exit(0)

    if mode == "error_exit":
        sys.stderr.write("boom: simulated failure\\n")
        sys.exit(int(os.environ.get("FAKE_CB_EXIT_CODE", "1")))

    if mode == "stdout_api_error_exit":
        # The real claude CLI prints transient API failures as PLAIN stdout
        # lines (not stream-json events, not stderr) before exiting non-zero.
        print("API Error: Connection closed mid-response", flush=True)
        sys.exit(1)

    if mode == "huge_line":
        emit({{
            "type": "system", "subtype": "init",
            "session_id": session_id, "model": "claude-sonnet-4-5",
        }})
        big = "x" * int(os.environ.get("FAKE_CB_HUGE_BYTES", str(256 * 1024)))
        emit({{
            "type": "assistant",
            "message": {{
                "role": "assistant", "model": "claude-sonnet-4-5",
                "content": [{{"type": "text", "text": big}}],
                "usage": {{"input_tokens": 10, "output_tokens": 5}},
            }},
        }})
        emit({{
            "type": "result", "subtype": "success", "is_error": False,
            "result": big, "session_id": session_id, "total_cost_usd": 0.002,
            "usage": {{
                "input_tokens": 10, "output_tokens": 5,
                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
            }},
        }})
        sys.exit(0)

    emit({{
        "type": "system", "subtype": "init",
        "session_id": session_id, "model": "claude-sonnet-4-5",
    }})

    if mode == "malformed":
        emit({{
            "type": "assistant",
            "message": {{"content": [{{"type": "text", "text": "partial only, no result"}}]}},
        }})
        sys.exit(0)

    if mode == "result_error":
        emit({{
            "type": "result", "subtype": "error", "is_error": True,
            "result": "simulated model error", "session_id": session_id,
        }})
        sys.exit(0)

    if mode in ("structured", "structured_always_invalid"):
        resumed = "--resume" in argv
        if mode == "structured" and resumed:
            text = '{{"passed": true, "notes": "ok"}}'
        else:
            text = "not json at all"
        emit({{
            "type": "assistant",
            "message": {{
                "role": "assistant", "model": "claude-sonnet-4-5",
                "content": [{{"type": "text", "text": text}}],
                "usage": {{"input_tokens": 10, "output_tokens": 5}},
            }},
        }})
        emit({{
            "type": "result", "subtype": "success", "is_error": False,
            "result": text, "session_id": session_id, "total_cost_usd": 0.002,
            "usage": {{
                "input_tokens": 10, "output_tokens": 5,
                "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
            }},
        }})
        sys.exit(0)

    # normal mode: a thinking block + a tool call/result + a final text result.
    emit({{
        "type": "assistant",
        "message": {{
            "role": "assistant", "model": "claude-sonnet-4-5",
            "content": [
                {{"type": "thinking", "thinking": "let's use a tool"}},
                {{
                    "type": "tool_use", "id": "tool-1", "name": "Bash",
                    "input": {{"command": "echo hi"}},
                }},
            ],
            "usage": {{"input_tokens": 20, "output_tokens": 8}},
        }},
    }})
    emit({{
        "type": "user",
        "message": {{"role": "user", "content": [
            {{"type": "tool_result", "tool_use_id": "tool-1", "content": "hi"}}
        ]}},
    }})
    emit({{
        "type": "assistant",
        "message": {{
            "role": "assistant", "model": "claude-sonnet-4-5",
            "content": [{{"type": "text", "text": "Done: hi"}}],
            "usage": {{"input_tokens": 25, "output_tokens": 12}},
        }},
    }})
    emit({{
        "type": "result", "subtype": "success", "is_error": False,
        "result": "Done: hi", "session_id": session_id, "total_cost_usd": 0.0035,
        "usage": {{
            "input_tokens": 25, "output_tokens": 12,
            "cache_read_input_tokens": 3, "cache_creation_input_tokens": 1,
        }},
    }})


if __name__ == "__main__":
    main()
"""


@pytest.fixture
def fake_cb(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Materialize the fake `cb` executable and point env vars at scratch files."""
    script = tmp_path / "cb"
    script.write_text(_FAKE_CB_SCRIPT, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    argv_dump = tmp_path / "argv_dump.ndjson"
    monkeypatch.setenv("FAKE_CB_ARGV_DUMP", str(argv_dump))
    monkeypatch.delenv("FAKE_CB_MODE", raising=False)
    monkeypatch.delenv("FAKE_CB_EXIT_CODE", raising=False)
    monkeypatch.delenv("FAKE_CB_LS_EXIT_CODE", raising=False)
    monkeypatch.delenv("FAKE_CB_SLEEP_SECONDS", raising=False)
    return script


def _read_argv_calls(tmp_path: Path) -> list[list[str]]:
    dump = tmp_path / "argv_dump.ndjson"
    if not dump.exists():
        return []
    return [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines() if line]


def _make_agent(**overrides: Any) -> AgentDef:
    fields: dict[str, Any] = {"name": "implementer", "prompt": "do the thing"}
    fields.update(overrides)
    return AgentDef(**fields)


class _EventRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, event_type: str, data: dict[str, Any]) -> None:
        self.events.append((event_type, data))


class TestExecuteNormalRun:
    async def test_parses_terminal_result_and_usage(self, fake_cb: Path, tmp_path: Path) -> None:
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()
        recorder = _EventRecorder()

        output = await provider.execute(
            agent,
            context={"box": "box-42"},
            rendered_prompt="do the thing",
            event_callback=recorder,
        )

        assert output.content == {"response": "Done: hi"}
        assert output.partial is False
        assert output.model == "claude-sonnet-4-5"
        assert output.input_tokens == 25
        assert output.output_tokens == 12
        assert output.tokens_used == 37
        assert output.cache_read_tokens == 3
        assert output.cache_write_tokens == 1
        assert output.raw_response["session_id"] == "sess-fake-1"
        assert output.raw_response["total_cost_usd"] == 0.0035

    async def test_emits_streaming_events_in_order(self, fake_cb: Path) -> None:
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()
        recorder = _EventRecorder()

        await provider.execute(
            agent,
            context={"box": "box-42"},
            rendered_prompt="do the thing",
            event_callback=recorder,
        )

        event_types = [e for e, _ in recorder.events]
        assert event_types[0] == "agent_turn_start"  # awaiting_model, pre-spawn
        assert "agent_reasoning" in event_types
        assert "agent_tool_start" in event_types
        assert "agent_tool_complete" in event_types
        assert "agent_message" in event_types
        # The tool_start/tool_complete pairing carried the right tool name.
        tool_start = next(d for e, d in recorder.events if e == "agent_tool_start")
        tool_complete = next(d for e, d in recorder.events if e == "agent_tool_complete")
        assert tool_start["tool_name"] == "Bash"
        assert tool_complete["tool_name"] == "Bash"
        assert tool_complete["result"] == "hi"

    async def test_argv_shape_matches_spec(self, fake_cb: Path, tmp_path: Path) -> None:
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent(name="my-role")

        await provider.execute(
            agent,
            context={"box": "box-42"},
            rendered_prompt="hello world",
        )

        [argv] = _read_argv_calls(tmp_path)
        assert argv == [
            "exec",
            "box-42",
            "claude",
            "-p",
            "hello world",
            "--agent",
            "my-role",
            "--model",
            "sonnet",
            "--permission-mode",
            "bypassPermissions",
            "--output-format",
            "stream-json",
            "--verbose",
        ]

    async def test_worktree_adds_workdir_flag(self, fake_cb: Path, tmp_path: Path) -> None:
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()

        await provider.execute(
            agent,
            context={"box": "box-42", "worktree": "/work/tree"},
            rendered_prompt="hello",
        )

        [argv] = _read_argv_calls(tmp_path)
        assert argv[:3] == ["exec", "--workdir", "/work/tree"]
        assert "box-42" in argv

    async def test_model_override_forwarded(self, fake_cb: Path, tmp_path: Path) -> None:
        provider = ClaudeboxProvider(cb_binary=str(fake_cb), model="haiku")
        agent = _make_agent(model="opus")

        await provider.execute(agent, context={"box": "b"}, rendered_prompt="p")

        [argv] = _read_argv_calls(tmp_path)
        assert argv[argv.index("--model") + 1] == "opus"


class TestContextAndToolsValidation:
    async def test_missing_box_raises_provider_error(self, fake_cb: Path) -> None:
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()

        with pytest.raises(ProviderError, match="requires a 'box' key"):
            await provider.execute(agent, context={}, rendered_prompt="p")

    async def test_box_resolved_from_workflow_input_fallback(
        self, fake_cb: Path, tmp_path: Path
    ) -> None:
        """M1b: `conductor run ... --input box=<id>` path (no top-level context key)."""
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()

        await provider.execute(
            agent,
            context={"workflow": {"input": {"box": "box-from-input"}}},
            rendered_prompt="hello",
        )

        [argv] = _read_argv_calls(tmp_path)
        assert "box-from-input" in argv

    async def test_worktree_resolved_from_workflow_input_fallback(
        self, fake_cb: Path, tmp_path: Path
    ) -> None:
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()

        await provider.execute(
            agent,
            context={"workflow": {"input": {"box": "b", "worktree": "/wt/from/input"}}},
            rendered_prompt="hello",
        )

        [argv] = _read_argv_calls(tmp_path)
        assert argv[:3] == ["exec", "--workdir", "/wt/from/input"]

    async def test_top_level_context_key_takes_precedence_over_workflow_input(
        self, fake_cb: Path, tmp_path: Path
    ) -> None:
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()

        await provider.execute(
            agent,
            context={"box": "direct-box", "workflow": {"input": {"box": "input-box"}}},
            rendered_prompt="hello",
        )

        [argv] = _read_argv_calls(tmp_path)
        assert "direct-box" in argv
        assert "input-box" not in argv

    async def test_missing_box_raises_even_with_unrelated_workflow_input(
        self, fake_cb: Path
    ) -> None:
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()

        with pytest.raises(ProviderError, match="requires a 'box' key"):
            await provider.execute(
                agent,
                context={"workflow": {"input": {"topic": "vector databases"}}},
                rendered_prompt="p",
            )

    async def test_nonempty_tools_allowlist_refused(self, fake_cb: Path) -> None:
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()

        with pytest.raises(ProviderError, match="does not support workflow tools allowlists"):
            await provider.execute(agent, context={"box": "b"}, rendered_prompt="p", tools=["Bash"])


class TestStructuredOutput:
    async def test_schema_instructions_injected_into_prompt(
        self, fake_cb: Path, tmp_path: Path
    ) -> None:
        os.environ["FAKE_CB_MODE"] = "structured"
        try:
            provider = ClaudeboxProvider(cb_binary=str(fake_cb))
            agent = _make_agent(output={"passed": OutputField(type="boolean")})

            await provider.execute(agent, context={"box": "b"}, rendered_prompt="verify it")

            calls = _read_argv_calls(tmp_path)
            first_prompt = calls[0][calls[0].index("-p") + 1]
            assert "verify it" in first_prompt
            assert "JSON Schema" in first_prompt
            assert '"passed"' in first_prompt
        finally:
            del os.environ["FAKE_CB_MODE"]

    async def test_recovers_via_resume_on_parse_failure(
        self, fake_cb: Path, tmp_path: Path
    ) -> None:
        os.environ["FAKE_CB_MODE"] = "structured"
        try:
            provider = ClaudeboxProvider(cb_binary=str(fake_cb))
            agent = _make_agent(output={"passed": OutputField(type="boolean")})

            output = await provider.execute(
                agent, context={"box": "b"}, rendered_prompt="verify it"
            )

            assert output.content == {"passed": True, "notes": "ok"}
            calls = _read_argv_calls(tmp_path)
            assert len(calls) == 2
            assert "--resume" not in calls[0]
            assert "--resume" in calls[1]
            assert calls[1][calls[1].index("--resume") + 1] == "sess-fake-1"
        finally:
            del os.environ["FAKE_CB_MODE"]

    async def test_exhausted_recovery_raises_validation_error(
        self, fake_cb: Path, tmp_path: Path
    ) -> None:
        os.environ["FAKE_CB_MODE"] = "structured_always_invalid"
        try:
            provider = ClaudeboxProvider(cb_binary=str(fake_cb))
            agent = _make_agent(
                output={"passed": OutputField(type="boolean")},
                retry={"max_parse_recovery_attempts": 1},
            )

            with pytest.raises(ValidationError, match="did not return parseable JSON"):
                await provider.execute(agent, context={"box": "b"}, rendered_prompt="verify it")

            # 1 initial attempt + 1 recovery attempt = 2 subprocess calls.
            assert len(_read_argv_calls(tmp_path)) == 2
        finally:
            del os.environ["FAKE_CB_MODE"]

    async def test_zero_recovery_attempts_fails_fast(self, fake_cb: Path, tmp_path: Path) -> None:
        os.environ["FAKE_CB_MODE"] = "structured_always_invalid"
        try:
            provider = ClaudeboxProvider(cb_binary=str(fake_cb))
            agent = _make_agent(
                output={"passed": OutputField(type="boolean")},
                retry={"max_parse_recovery_attempts": 0},
            )

            with pytest.raises(ValidationError):
                await provider.execute(agent, context={"box": "b"}, rendered_prompt="verify it")

            assert len(_read_argv_calls(tmp_path)) == 1
        finally:
            del os.environ["FAKE_CB_MODE"]


class TestErrorMapping:
    async def test_nonzero_exit_raises_provider_error(self, fake_cb: Path) -> None:
        os.environ["FAKE_CB_MODE"] = "error_exit"
        os.environ["FAKE_CB_EXIT_CODE"] = "1"
        try:
            provider = ClaudeboxProvider(cb_binary=str(fake_cb))
            agent = _make_agent()
            with pytest.raises(ProviderError, match="exited with code 1"):
                await provider.execute(agent, context={"box": "b"}, rendered_prompt="p")
        finally:
            del os.environ["FAKE_CB_MODE"]
            del os.environ["FAKE_CB_EXIT_CODE"]

    async def test_malformed_stream_missing_result_raises_provider_error(
        self, fake_cb: Path
    ) -> None:
        os.environ["FAKE_CB_MODE"] = "malformed"
        try:
            provider = ClaudeboxProvider(cb_binary=str(fake_cb))
            agent = _make_agent()
            with pytest.raises(ProviderError, match="never produced a terminal"):
                await provider.execute(agent, context={"box": "b"}, rendered_prompt="p")
        finally:
            del os.environ["FAKE_CB_MODE"]

    async def test_result_is_error_raises_provider_error(self, fake_cb: Path) -> None:
        os.environ["FAKE_CB_MODE"] = "result_error"
        try:
            provider = ClaudeboxProvider(cb_binary=str(fake_cb))
            agent = _make_agent()
            with pytest.raises(ProviderError, match="simulated model error"):
                await provider.execute(agent, context={"box": "b"}, rendered_prompt="p")
        finally:
            del os.environ["FAKE_CB_MODE"]

    async def test_stdout_api_error_classifies_retryable(
        self, fake_cb: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: `API Error: Connection closed mid-response` arrives as a
        plain stdout line with EMPTY stderr; a stderr-only classification called
        this transient failure fatal and killed a 6/7-milestones live run."""
        monkeypatch.setenv("FAKE_CB_MODE", "stdout_api_error_exit")
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()
        with pytest.raises(ProviderError) as excinfo:
            await provider.execute(agent, context={"box": "b"}, rendered_prompt="p")
        assert excinfo.value.is_retryable is True
        assert "API Error: Connection closed" in str(excinfo.value)

    async def test_stderr_failure_without_transient_signal_stays_non_retryable(
        self, fake_cb: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("FAKE_CB_MODE", "error_exit")
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()
        with pytest.raises(ProviderError) as excinfo:
            await provider.execute(agent, context={"box": "b"}, rendered_prompt="p")
        assert excinfo.value.is_retryable is False

    async def test_cb_binary_missing_raises_provider_error(self, tmp_path: Path) -> None:
        provider = ClaudeboxProvider(cb_binary=str(tmp_path / "no-such-cb-binary"))
        agent = _make_agent()
        with pytest.raises(ProviderError, match="claudebox CLI not found"):
            await provider.execute(agent, context={"box": "b"}, rendered_prompt="p")


class TestInterrupt:
    async def test_interrupt_terminates_subprocess_and_returns_partial(self, fake_cb: Path) -> None:
        os.environ["FAKE_CB_MODE"] = "sleep"
        os.environ["FAKE_CB_SLEEP_SECONDS"] = "30"
        try:
            provider = ClaudeboxProvider(cb_binary=str(fake_cb))
            agent = _make_agent()
            interrupt_signal = asyncio.Event()

            async def _fire_interrupt() -> None:
                await asyncio.sleep(0.3)
                interrupt_signal.set()

            fire_task = asyncio.ensure_future(_fire_interrupt())
            import time as _time

            start = _time.monotonic()
            output = await provider.execute(
                agent,
                context={"box": "b"},
                rendered_prompt="p",
                interrupt_signal=interrupt_signal,
            )
            elapsed = _time.monotonic() - start
            await fire_task

            assert output.partial is True
            # Proves the subprocess was actually terminated rather than the
            # test waiting out the full 30s sleep.
            assert elapsed < 10
        finally:
            del os.environ["FAKE_CB_MODE"]
            del os.environ["FAKE_CB_SLEEP_SECONDS"]


class TestStreamLimit:
    async def test_streams_single_line_larger_than_64kib(
        self, fake_cb: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression: `claude --output-format stream-json` emits one JSON
        object per line; a single large tool_result/file-write body exceeded
        asyncio's default 64 KiB StreamReader limit, so `readline()` raised
        `ValueError: Separator is found, but chunk is longer than limit` and
        killed the workflow mid-milestone."""
        monkeypatch.setenv("FAKE_CB_MODE", "huge_line")
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()
        output = await provider.execute(agent, context={"box": "b"}, rendered_prompt="p")
        assert len(output.content["response"]) > 64 * 1024


class TestValidateConnection:
    async def test_true_when_cb_ls_succeeds(self, fake_cb: Path) -> None:
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        assert await provider.validate_connection() is True

    async def test_false_when_cb_ls_fails(self, fake_cb: Path) -> None:
        os.environ["FAKE_CB_LS_EXIT_CODE"] = "1"
        try:
            provider = ClaudeboxProvider(cb_binary=str(fake_cb))
            assert await provider.validate_connection() is False
        finally:
            del os.environ["FAKE_CB_LS_EXIT_CODE"]

    async def test_false_when_cb_binary_missing(self, tmp_path: Path) -> None:
        provider = ClaudeboxProvider(cb_binary=str(tmp_path / "no-such-cb"))
        assert await provider.validate_connection() is False


class TestClose:
    async def test_close_is_a_no_op(self, fake_cb: Path) -> None:
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        await provider.close()  # must not raise
