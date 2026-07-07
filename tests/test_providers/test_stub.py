"""Tests for StubProvider — the scripted, no-LLM/no-box/no-network test double."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from conductor.config.schema import AgentDef
from conductor.exceptions import ProviderError
from conductor.providers.stub import StubProvider

pytestmark = pytest.mark.asyncio


def _make_agent(name: str = "verifier") -> AgentDef:
    return AgentDef(name=name, prompt="do the thing")


def _write_script(tmp_path: Path, script: dict[str, Any]) -> str:
    path = tmp_path / "script.json"
    path.write_text(json.dumps(script), encoding="utf-8")
    return str(path)


class _EventRecorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, event_type: str, data: dict[str, Any]) -> None:
        self.events.append((event_type, data))


class TestScriptedSequenceAdvances:
    async def test_two_call_sequence_advances_fail_then_pass(self, tmp_path: Path) -> None:
        script_path = _write_script(
            tmp_path,
            {
                "steps": {
                    "verifier": [
                        {"content": {"passed": False, "notes": "missing tests"}},
                        {"content": {"passed": True}},
                    ]
                }
            },
        )
        provider = StubProvider(script_path=script_path)
        agent = _make_agent("verifier")
        recorder = _EventRecorder()

        first = await provider.execute(agent, {}, "prompt", event_callback=recorder)
        second = await provider.execute(agent, {}, "prompt", event_callback=recorder)

        assert first.content == {"passed": False, "notes": "missing tests"}
        assert second.content == {"passed": True}
        assert len(recorder.events) >= 2  # at least one event pair per call

    async def test_exhausted_sequence_repeats_last_entry(self, tmp_path: Path) -> None:
        script_path = _write_script(
            tmp_path,
            {"steps": {"verifier": [{"content": {"n": 1}}, {"content": {"n": 2}}]}},
        )
        provider = StubProvider(script_path=script_path)
        agent = _make_agent("verifier")

        await provider.execute(agent, {}, "p")
        await provider.execute(agent, {}, "p")
        third = await provider.execute(agent, {}, "p")
        fourth = await provider.execute(agent, {}, "p")

        assert third.content == {"n": 2}
        assert fourth.content == {"n": 2}

    async def test_independent_cursors_per_step_name(self, tmp_path: Path) -> None:
        script_path = _write_script(
            tmp_path,
            {
                "steps": {
                    "implementer": [{"content": {"who": "impl-1"}}, {"content": {"who": "impl-2"}}],
                    "verifier": [{"content": {"who": "verify-1"}}],
                }
            },
        )
        provider = StubProvider(script_path=script_path)

        impl1 = await provider.execute(_make_agent("implementer"), {}, "p")
        verify1 = await provider.execute(_make_agent("verifier"), {}, "p")
        impl2 = await provider.execute(_make_agent("implementer"), {}, "p")

        assert impl1.content == {"who": "impl-1"}
        assert verify1.content == {"who": "verify-1"}
        assert impl2.content == {"who": "impl-2"}


class TestDefaultAndMissingEntries:
    async def test_falls_back_to_default_for_unscripted_step(self, tmp_path: Path) -> None:
        script_path = _write_script(
            tmp_path,
            {"steps": {}, "default": {"content": {"response": "fallback"}}},
        )
        provider = StubProvider(script_path=script_path)

        output = await provider.execute(_make_agent("mystery-step"), {}, "p")

        assert output.content == {"response": "fallback"}

    async def test_missing_step_and_no_default_raises_provider_error(self, tmp_path: Path) -> None:
        script_path = _write_script(tmp_path, {"steps": {}})
        provider = StubProvider(script_path=script_path)

        with pytest.raises(ProviderError, match="No scripted output"):
            await provider.execute(_make_agent("mystery-step"), {}, "p")


class TestErrorEntries:
    async def test_error_entry_raises_provider_error(self, tmp_path: Path) -> None:
        script_path = _write_script(
            tmp_path,
            {"steps": {"implementer": [{"error": "simulated failure", "error_retryable": True}]}},
        )
        provider = StubProvider(script_path=script_path)

        with pytest.raises(ProviderError, match="simulated failure") as exc_info:
            await provider.execute(_make_agent("implementer"), {}, "p")
        assert exc_info.value.is_retryable is True

    async def test_entry_missing_content_and_error_raises(self, tmp_path: Path) -> None:
        script_path = _write_script(tmp_path, {"steps": {"implementer": [{}]}})
        provider = StubProvider(script_path=script_path)

        with pytest.raises(ProviderError, match="neither 'content' nor 'error'"):
            await provider.execute(_make_agent("implementer"), {}, "p")


class TestUsageAndModelFields:
    async def test_forwards_token_and_model_fields(self, tmp_path: Path) -> None:
        script_path = _write_script(
            tmp_path,
            {
                "steps": {
                    "implementer": [
                        {
                            "content": {"diff": "x"},
                            "input_tokens": 100,
                            "output_tokens": 50,
                            "tokens_used": 150,
                            "cache_read_tokens": 10,
                            "cache_write_tokens": 5,
                            "model": "stub-model-v1",
                        }
                    ]
                }
            },
        )
        provider = StubProvider(script_path=script_path)

        output = await provider.execute(_make_agent("implementer"), {}, "p")

        assert output.input_tokens == 100
        assert output.output_tokens == 50
        assert output.tokens_used == 150
        assert output.cache_read_tokens == 10
        assert output.cache_write_tokens == 5
        assert output.model == "stub-model-v1"


class TestEvents:
    async def test_custom_events_are_emitted_verbatim(self, tmp_path: Path) -> None:
        script_path = _write_script(
            tmp_path,
            {
                "steps": {
                    "implementer": [
                        {
                            "content": {"ok": True},
                            "events": [
                                ["agent_tool_start", {"tool_name": "Bash"}],
                                ["agent_tool_complete", {"tool_name": "Bash", "result": "done"}],
                            ],
                        }
                    ]
                }
            },
        )
        provider = StubProvider(script_path=script_path)
        recorder = _EventRecorder()

        await provider.execute(_make_agent("implementer"), {}, "p", event_callback=recorder)

        assert recorder.events == [
            ("agent_tool_start", {"tool_name": "Bash"}),
            ("agent_tool_complete", {"tool_name": "Bash", "result": "done"}),
        ]

    async def test_default_events_emitted_when_none_scripted(self, tmp_path: Path) -> None:
        script_path = _write_script(
            tmp_path, {"steps": {"implementer": [{"content": {"ok": True}}]}}
        )
        provider = StubProvider(script_path=script_path)
        recorder = _EventRecorder()

        await provider.execute(_make_agent("implementer"), {}, "p", event_callback=recorder)

        assert [e for e, _ in recorder.events] == ["agent_turn_start", "agent_message"]


class TestInterrupt:
    async def test_interrupt_signal_already_set_returns_partial(self, tmp_path: Path) -> None:
        script_path = _write_script(
            tmp_path, {"steps": {"implementer": [{"content": {"ok": True}}]}}
        )
        provider = StubProvider(script_path=script_path)
        interrupt_signal = asyncio.Event()
        interrupt_signal.set()

        output = await provider.execute(
            _make_agent("implementer"), {}, "p", interrupt_signal=interrupt_signal
        )

        assert output.partial is True
        assert output.content == {"ok": True}

    async def test_scripted_partial_flag_is_forwarded(self, tmp_path: Path) -> None:
        script_path = _write_script(
            tmp_path,
            {"steps": {"implementer": [{"content": {"ok": True}, "partial": True}]}},
        )
        provider = StubProvider(script_path=script_path)

        output = await provider.execute(_make_agent("implementer"), {}, "p")

        assert output.partial is True


class TestScriptResolution:
    async def test_env_var_fallback_when_no_script_path_given(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script_path = _write_script(
            tmp_path, {"steps": {"implementer": [{"content": {"via": "env"}}]}}
        )
        monkeypatch.setenv("CONDUCTOR_STUB_SCRIPT", script_path)
        provider = StubProvider()

        output = await provider.execute(_make_agent("implementer"), {}, "p")

        assert output.content == {"via": "env"}

    async def test_no_script_configured_raises_on_execute_not_construction(self) -> None:
        provider = StubProvider()  # must not raise here
        with pytest.raises(ProviderError, match="no script configured"):
            await provider.execute(_make_agent("implementer"), {}, "p")

    async def test_missing_script_file_raises_provider_error(self, tmp_path: Path) -> None:
        provider = StubProvider(script_path=str(tmp_path / "does-not-exist.json"))
        with pytest.raises(ProviderError, match="Could not read stub script"):
            await provider.execute(_make_agent("implementer"), {}, "p")

    async def test_invalid_json_script_raises_provider_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        provider = StubProvider(script_path=str(bad))
        with pytest.raises(ProviderError, match="not valid JSON"):
            await provider.execute(_make_agent("implementer"), {}, "p")

    async def test_non_object_script_raises_provider_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("[1, 2, 3]", encoding="utf-8")
        provider = StubProvider(script_path=str(bad))
        with pytest.raises(ProviderError, match="must be a JSON object"):
            await provider.execute(_make_agent("implementer"), {}, "p")


class TestProviderLifecycle:
    async def test_validate_connection_always_true(self) -> None:
        provider = StubProvider()
        assert await provider.validate_connection() is True

    async def test_close_is_a_no_op(self) -> None:
        provider = StubProvider()
        await provider.close()  # must not raise
