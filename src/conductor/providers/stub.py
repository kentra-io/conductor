"""Stub provider — a scripted test double for exercising workflow control-flow.

Registered as provider type ``"stub"``. Returns pre-scripted
:class:`~conductor.providers.base.AgentOutput` sequences keyed by
step/agent name, so a workflow's control-flow (retry/escalation ladders,
routes, human gates, resume) can be tested end-to-end with **no LLM, no
claudebox, no network**.

Script file format
------------------

A JSON file with the shape::

    {
      "steps": {
        "<agent_name>": [
          { <scripted AgentOutput fields...> },
          { <scripted AgentOutput fields...> }
        ],
        "...": [ ... ]
      },
      "default": { <scripted AgentOutput fields...> }
    }

``steps`` maps an agent/step name (``AgentDef.name`` -- the same string used
as the workflow YAML step's ``name:``) to an **ordered list** of scripted
entries. Each call to :meth:`StubProvider.execute` for that step name
advances to the next entry in its list -- so a step scripted with
``[fail, fail, pass]`` fails its first two calls and passes on the third,
which is exactly what's needed to test a bounded retry/escalation ladder.
Once a step's list is exhausted, the **last** entry is returned on every
subsequent call (a "settled" steady state) rather than raising, so a
workflow that calls a step more times than scripted doesn't explode.

A step name absent from ``steps`` falls back to the top-level ``default``
entry, if present; if neither exists, ``execute()`` raises
:class:`~conductor.exceptions.ProviderError` naming the missing step and the
script path, so an incomplete script fails loudly rather than returning
silently-wrong data.

Each scripted entry is a dict with these keys (all optional except
``content``, unless ``error`` is set):

* ``content`` (dict, **required** unless ``error`` is set) -- becomes
  ``AgentOutput.content`` verbatim.
* ``raw_response`` (any, default ``content``) -- ``AgentOutput.raw_response``.
* ``tokens_used`` / ``input_tokens`` / ``output_tokens`` /
  ``cache_read_tokens`` / ``cache_write_tokens`` (int) -- forwarded as-is.
* ``model`` (str) -- forwarded as-is.
* ``partial`` (bool, default ``false``) -- forwarded as-is; also
  automatically forced ``true`` (with the same content) if
  ``interrupt_signal`` is already set when this step is called.
* ``error`` (str) -- when present, ``execute()`` raises
  :class:`~conductor.exceptions.ProviderError` with this message instead of
  returning an output. ``error_retryable`` (bool, default ``false``)
  controls the raised error's ``is_retryable``.
* ``events`` (list of ``[event_type, data]`` pairs) -- emitted via
  ``event_callback`` (in order) before the entry is returned/raised. When
  omitted, a plausible default pair (``agent_turn_start`` +
  ``agent_message`` echoing a short content preview) is emitted instead, so
  dashboards/JSONL consumers always see *something* for a stubbed step.

Example script::

    {
      "steps": {
        "implementer": [
          {"content": {"diff": "stub-diff-1"}, "cost_usd_hint": 0.01},
          {"content": {"diff": "stub-diff-2"}}
        ],
        "verifier": [
          {"content": {"passed": false, "notes": "missing tests"}},
          {"content": {"passed": true}}
        ]
      },
      "default": {"content": {"response": "unscripted stub default"}}
    }

The script path is resolved at construction time from the ``script_path``
constructor argument (wired from ``ProviderSettings.stub_script_path`` by
the factory), falling back to the ``CONDUCTOR_STUB_SCRIPT`` environment
variable. Neither set is an error at construction time -- it only surfaces
when :meth:`execute` is actually called, so ``conductor validate`` and
``get_capabilities()`` never require a script to exist.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from conductor.exceptions import ProviderError
from conductor.providers.base import AgentOutput, AgentProvider, EventCallback
from conductor.providers.capabilities import ProviderCapabilities

if TYPE_CHECKING:
    from conductor.config.schema import AgentDef

logger = logging.getLogger(__name__)

_SCRIPT_PATH_ENV_VAR: Final[str] = "CONDUCTOR_STUB_SCRIPT"


def _safe_callback(callback: EventCallback, event_type: str, data: dict[str, Any]) -> None:
    try:
        callback(event_type, data)
    except Exception:
        logger.debug("Error in event_callback for %s", event_type, exc_info=True)


class StubProvider(AgentProvider):
    """Scripted test-double provider -- no LLM, no box, no network.

    See the module docstring for the exact script-file format.
    """

    CAPABILITIES = ProviderCapabilities(
        tier="experimental",
        mcp_tools=False,
        workflow_tools_passthrough=False,
        # Emits a plausible agent_turn_start/agent_message/agent_tool_*
        # pair (or the script's own `events`) per call.
        streaming_events=True,
        agent_reasoning_events=False,
        reasoning_effort=None,
        # The scripted `content` dict is returned verbatim -- by
        # construction it "matches" whatever schema the script author
        # wrote it to match.
        structured_output="native",
        # If interrupt_signal is already set when a step is called, the
        # scripted content is returned with partial=True immediately.
        interrupt=True,
        max_session_seconds=False,
        checkpoint_resume=False,
        # Scripted entries may carry token counts; forwarded verbatim.
        usage_tracking=True,
        concurrent_safe=True,
        upstream_pin=None,
        maintainer="kentra (M1 test double)",
    )

    def __init__(self, script_path: str | None = None) -> None:
        """Initialize the stub provider.

        Args:
            script_path: Path to the JSON script file. Falls back to the
                ``CONDUCTOR_STUB_SCRIPT`` env var when omitted. Not required
                at construction time -- resolution/loading is deferred to
                the first :meth:`execute` call so a script-less StubProvider
                can still be constructed and validated.
        """
        self._script_path = script_path or os.environ.get(_SCRIPT_PATH_ENV_VAR)
        self._script: dict[str, Any] | None = None
        self._call_counts: dict[str, int] = {}

    async def execute(
        self,
        agent: AgentDef,
        context: dict[str, Any],
        rendered_prompt: str,
        tools: list[str] | None = None,
        interrupt_signal: asyncio.Event | None = None,
        event_callback: EventCallback | None = None,
    ) -> AgentOutput:
        del context, rendered_prompt, tools  # unused -- pure script lookup

        script = self._load_script()
        step_name = agent.name
        steps = script.get("steps", {})
        entry = self._resolve_entry(step_name, steps, script.get("default"))

        events = entry.get("events")
        if events:
            for event_type, data in events:
                if event_callback:
                    _safe_callback(event_callback, event_type, data)
        elif event_callback:
            _safe_callback(event_callback, "agent_turn_start", {"turn": 1})
            preview = json.dumps(entry.get("content", {}))[:200]
            _safe_callback(event_callback, "agent_message", {"content": preview})

        self._call_counts[step_name] = self._call_counts.get(step_name, 0) + 1

        if "error" in entry:
            raise ProviderError(
                str(entry["error"]),
                is_retryable=bool(entry.get("error_retryable", False)),
            )

        if "content" not in entry:
            raise ProviderError(
                f"Scripted entry for step {step_name!r} in {self._script_path!r} has "
                "neither 'content' nor 'error' -- every non-error entry must "
                "declare 'content'."
            )

        partial = bool(entry.get("partial", False))
        if interrupt_signal is not None and interrupt_signal.is_set():
            partial = True

        return AgentOutput(
            content=entry["content"],
            raw_response=entry.get("raw_response", entry["content"]),
            tokens_used=entry.get("tokens_used"),
            input_tokens=entry.get("input_tokens"),
            output_tokens=entry.get("output_tokens"),
            cache_read_tokens=entry.get("cache_read_tokens"),
            cache_write_tokens=entry.get("cache_write_tokens"),
            model=entry.get("model"),
            partial=partial,
        )

    async def validate_connection(self) -> bool:
        """Always ``True`` -- there is no backend to reach."""
        return True

    async def close(self) -> None:
        """No-op -- the stub holds no resources to release."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_script(self) -> dict[str, Any]:
        """Load and cache the script file. Raises loudly if missing/invalid."""
        if self._script is not None:
            return self._script
        if not self._script_path:
            raise ProviderError(
                "StubProvider has no script configured -- set "
                "`provider.stub_script_path` in the workflow YAML or the "
                f"{_SCRIPT_PATH_ENV_VAR} environment variable.",
                suggestion=("provider:\n  name: stub\n  stub_script_path: path/to/script.json"),
            )
        path = Path(self._script_path)
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ProviderError(
                f"Could not read stub script {self._script_path!r}: {exc}",
            ) from exc
        try:
            script = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"Stub script {self._script_path!r} is not valid JSON: {exc}",
            ) from exc
        if not isinstance(script, dict):
            raise ProviderError(
                f"Stub script {self._script_path!r} must be a JSON object with a "
                "'steps' key, got a top-level "
                f"{type(script).__name__}."
            )
        self._script = script
        return script

    def _resolve_entry(
        self,
        step_name: str,
        steps: dict[str, Any],
        default: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Pick the scripted entry for this call, advancing the per-step cursor.

        Repeated calls to the same step advance through its list; once
        exhausted, the last entry repeats (a settled steady state).
        """
        outputs = steps.get(step_name)
        if outputs:
            call_index = self._call_counts.get(step_name, 0)
            idx = min(call_index, len(outputs) - 1)
            entry = outputs[idx]
            if not isinstance(entry, dict):
                raise ProviderError(
                    f"Scripted entry #{idx} for step {step_name!r} in "
                    f"{self._script_path!r} must be a JSON object, got "
                    f"{type(entry).__name__}."
                )
            return entry
        if default is not None:
            return default
        raise ProviderError(
            f"No scripted output for step {step_name!r} and no top-level "
            f"'default' entry in stub script {self._script_path!r}.",
            suggestion=(
                f"Add a \"{step_name}\": [...] entry to the script's 'steps', "
                "or add a top-level 'default' entry."
            ),
        )
