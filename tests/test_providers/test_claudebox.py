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
from conductor.providers.claudebox import (
    ClaudeboxProvider,
    _classify_retryable,
    _nonzero_exit_detail,
    _RunOutcome,
)

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

    if mode == "stderr_benign_plus_stdout_connection_exit":
        # Non-empty stderr with NO transient signal, plus a stdout noise line
        # that DOES carry one. Classification must still see the noise tail
        # even though stderr alone would otherwise satisfy `detail`.
        print("API Error: Connection closed mid-response", flush=True)
        sys.stderr.write("boom: simulated failure\\n")
        sys.exit(1)

    if mode == "stdout_oauth_and_connection_exit":
        # OAuth text and a retryable keyword in the SAME stdout noise line;
        # the OAuth check must still win.
        print("OAuth session expired ... connection", flush=True)
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

    if mode == "oauth_content_nonzero_exit":
        # The exact incident shape (kentra-io/harness#3): a dead box's OAuth
        # failure never reaches stderr or a terminal `result` event — it's
        # the text of the last assistant message, immediately followed by a
        # non-zero exit. Empty stderr, no stdout noise.
        emit({{
            "type": "assistant",
            "message": {{
                "role": "assistant", "model": "claude-sonnet-4-5",
                "content": [{{
                    "type": "text",
                    "text": (
                        "Failed to authenticate: OAuth session expired "
                        "and could not be refreshed"
                    ),
                }}],
                "usage": {{"input_tokens": 10, "output_tokens": 5}},
            }},
        }})
        sys.exit(1)

    if mode == "stdout_connection_noise_oauth_content_exit":
        # Cross-source classification: a retryable-looking keyword in stdout
        # NOISE (not stream-json) alongside an OAuth-expiry message in the
        # agent's stream-json CONTENT, then a non-zero exit. Both sources
        # feed one `classify_text` string, so the OAuth check must still win
        # even though the noise line alone would classify retryable.
        print("API Error: Connection closed mid-response", flush=True)
        emit({{
            "type": "assistant",
            "message": {{
                "role": "assistant", "model": "claude-sonnet-4-5",
                "content": [{{
                    "type": "text",
                    "text": (
                        "Failed to authenticate: OAuth session expired "
                        "and could not be refreshed"
                    ),
                }}],
                "usage": {{"input_tokens": 10, "output_tokens": 5}},
            }},
        }})
        sys.exit(1)

    if mode == "result_error":
        emit({{
            "type": "result", "subtype": "error", "is_error": True,
            "result": "simulated model error", "session_id": session_id,
        }})
        sys.exit(0)

    if mode == "result_error_retryable_nonzero_exit":
        # Edge case: a terminal `result` event carries a retryable signal in
        # result_error_message, AND the process also exits non-zero (so the
        # exit_code!=0 branch preempts the result_is_error branch). Stderr
        # and stdout noise are both empty — classification must still see
        # result_error_message, mirroring ab0ff4c's original `diag`.
        emit({{
            "type": "result", "subtype": "error", "is_error": True,
            "result": "API Error: 429 rate limit exceeded", "session_id": session_id,
        }})
        sys.exit(1)

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
    monkeypatch.delenv("CLAUDE_CODE_LONG_LIVED_TOKEN", raising=False)
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


class TestLongLivedTokenEnvInjection:
    """kentra-io/harness#3: the orchestration daemon holds a 1-year
    non-rotating `CLAUDE_CODE_LONG_LIVED_TOKEN` (macOS keychain); the
    provider maps it to `CLAUDE_CODE_OAUTH_TOKEN` per invocation, forwarded
    into the `cb exec` subprocess env plus a bare-name `-e` flag so `docker
    exec` carries it into the box -- the secret itself never enters argv."""

    async def test_build_env_maps_long_lived_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CLAUDE_CODE_LONG_LIVED_TOKEN", "sk-ant-oat01-x")
        provider = ClaudeboxProvider()
        env = provider._build_env()
        assert env is not None
        assert env["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-x"

    async def test_build_env_still_none_without_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CLAUDE_CODE_LONG_LIVED_TOKEN", raising=False)
        provider = ClaudeboxProvider()
        assert provider._build_env() is None

    async def test_build_argv_forwards_oauth_env_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDE_CODE_LONG_LIVED_TOKEN", "sk-ant-oat01-x")
        provider = ClaudeboxProvider()
        argv = provider._build_argv("box", None, "implementer", "opus", "hi")
        i = argv.index("-e")
        assert argv[i + 1] == "CLAUDE_CODE_OAUTH_TOKEN"  # bare name -- no secret in argv
        assert "sk-ant-oat01-x" not in " ".join(argv)

    async def test_build_argv_no_env_flag_without_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("CLAUDE_CODE_LONG_LIVED_TOKEN", raising=False)
        provider = ClaudeboxProvider()
        assert "-e" not in provider._build_argv("box", None, "implementer", "opus", "hi")


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

    async def test_nonempty_stderr_still_classifies_via_stdout_noise_tail(
        self, fake_cb: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guards against narrowing classification to `detail`'s fallback chain:
        stderr is non-empty (so `detail` resolves from stderr alone) but carries
        no transient signal itself — the retryable keyword only appears in a
        stdout noise line. Classification must still see it (ab0ff4c's
        stdout-aware fix must survive alongside the OAuth check)."""
        monkeypatch.setenv("FAKE_CB_MODE", "stderr_benign_plus_stdout_connection_exit")
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()
        with pytest.raises(ProviderError) as excinfo:
            await provider.execute(agent, context={"box": "b"}, rendered_prompt="p")
        assert excinfo.value.is_retryable is True

    async def test_oauth_in_stdout_noise_wins_over_connection_keyword(
        self, fake_cb: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inverse guard: OAuth text anywhere in the combined classification
        input (here, a stdout noise line) must still classify non-retryable
        even when "connection" appears in that same line."""
        monkeypatch.setenv("FAKE_CB_MODE", "stdout_oauth_and_connection_exit")
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()
        with pytest.raises(ProviderError) as excinfo:
            await provider.execute(agent, context={"box": "b"}, rendered_prompt="p")
        assert excinfo.value.is_retryable is False

    async def test_result_error_message_still_feeds_classification_on_nonzero_exit(
        self, fake_cb: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guards against narrowing classification away from ab0ff4c's
        original `diag` inputs: a terminal `result` event carries a
        retryable signal in `result_error_message`, but the process ALSO
        exits non-zero — so the exit_code!=0 branch (not the result_is_error
        branch) is what raises. Stderr and stdout noise are both empty;
        classification must still see result_error_message."""
        monkeypatch.setenv("FAKE_CB_MODE", "result_error_retryable_nonzero_exit")
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()
        with pytest.raises(ProviderError) as excinfo:
            await provider.execute(agent, context={"box": "b"}, rendered_prompt="p")
        assert excinfo.value.is_retryable is True

    async def test_oauth_content_nonzero_exit_end_to_end(
        self, fake_cb: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The incident's exact shape (kentra-io/harness#3): an assistant
        message carries the OAuth failure as text content, immediately
        followed by a non-zero exit — no stderr, no stdout noise, no
        terminal `result` event. Proves the fix end-to-end: the content
        tail surfaces in the error message AND classifies non-retryable."""
        monkeypatch.setenv("FAKE_CB_MODE", "oauth_content_nonzero_exit")
        provider = ClaudeboxProvider(cb_binary=str(fake_cb))
        agent = _make_agent()
        with pytest.raises(ProviderError) as excinfo:
            await provider.execute(agent, context={"box": "b"}, rendered_prompt="p")
        assert excinfo.value.is_retryable is False
        assert "OAuth session expired" in str(excinfo.value)

    async def test_cross_source_oauth_content_wins_over_stdout_noise_keyword(
        self, fake_cb: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cross-source classification discrimination: a retryable-looking
        keyword arrives via stdout NOISE while the OAuth-expiry text arrives
        via stream-json CONTENT -- two different sources feeding the same
        `classify_text` string. The OAuth check must still win even though
        neither source alone would be ambiguous; pins that `content_tail` is
        actually wired into classification (a reviewer noted this flips to
        retryable if content is dropped from `classify_text`)."""
        monkeypatch.setenv("FAKE_CB_MODE", "stdout_connection_noise_oauth_content_exit")
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


class TestClassifyRetryableOAuth:
    """Regression: a dead box's OAuth session-expiry surfaced as agent-message
    stream *content* (stdout JSON), not stderr, so it fell through to the
    generic retryable default and four stacked retry layers churned for ~52
    minutes on an unrecoverable failure (kentra-io/harness#3)."""

    async def test_oauth_expired_not_retryable(self) -> None:
        assert (
            _classify_retryable(
                "Failed to authenticate: OAuth session expired and could not be refreshed", 1
            )
            is False
        )
        assert _classify_retryable("OAuth token has expired", 1) is False
        assert _classify_retryable("credentials could not be refreshed", 1) is False
        assert (
            _classify_retryable(
                "Failed to authenticate. API Error: 401 OAuth access token is invalid.", 1
            )
            is False
        )

    async def test_oauth_expired_wins_over_retryable_connection_keyword(self) -> None:
        """`"connection"` alone classifies retryable; the OAuth check must win
        even when a retryable keyword also appears in the same message."""
        assert _classify_retryable("OAuth session expired ... connection", 1) is False

    async def test_nonzero_exit_detail_includes_content_parts(self) -> None:
        """When stderr and stdout noise are both empty, the only diagnostic is
        agent text content — the incident shape — so it must surface."""
        outcome = _RunOutcome()
        outcome.content_parts.append(
            "Failed to authenticate: OAuth session expired and could not be refreshed"
        )
        detail = _nonzero_exit_detail("", outcome)
        assert "OAuth session expired" in detail


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


class TestStallWatchdog:
    async def test_stall_kills_subprocess_and_raises_retryable(self, tmp_path: Path) -> None:
        """No stdout for longer than the threshold -> retryable ProviderError."""
        script = tmp_path / "cb"
        script.write_text(
            "#!/bin/bash\n"
            'echo \'{"type":"system","subtype":"init","session_id":"s1","model":"m"}\'\n'
            "sleep 30\n"
        )
        script.chmod(0o755)
        provider = ClaudeboxProvider(cb_binary=str(script), stall_timeout_seconds=0.5)
        agent = _make_agent()
        with pytest.raises(ProviderError, match="stall") as exc_info:
            await provider.execute(
                agent, context={"box": "b", "worktree": str(tmp_path)}, rendered_prompt="p"
            )
        assert exc_info.value.is_retryable is True

    def test_zero_threshold_disables_watchdog(self) -> None:
        provider = ClaudeboxProvider(cb_binary="cb", stall_timeout_seconds=0)
        assert provider._stall_timeout is None

    def test_env_var_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CONDUCTOR_CLAUDEBOX_STALL_SECONDS", "120")
        provider = ClaudeboxProvider(cb_binary="cb")
        assert provider._stall_timeout == 120.0

    def test_builtin_default_is_600(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CONDUCTOR_CLAUDEBOX_STALL_SECONDS", raising=False)
        provider = ClaudeboxProvider(cb_binary="cb")
        assert provider._stall_timeout == 600.0
