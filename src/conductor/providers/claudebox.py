"""Claudebox provider — drives a ``claude`` CLI subprocess inside a claudebox sandbox.

M1 scope: the real subprocess-driving :meth:`ClaudeboxProvider.execute`. Each
call spawns ``cb exec <box> claude -p <prompt> --agent <role> --model <model>
--permission-mode bypassPermissions --output-format stream-json --verbose``,
streams the ``stream-json`` lines, and normalizes the terminal ``result``
event into an :class:`~conductor.providers.base.AgentOutput`.

Context contract (the keys this provider reads from the workflow ``context``
dict passed into :meth:`execute`):

* ``context["box"]`` (**required**, str) — the claudebox box/container
  identifier that ``cb exec <box>`` targets. Box lifecycle (``cb run``,
  worktree + overlay mounts) happens *outside* this provider — a launcher
  script step or the embedding service sets this key before any
  claudebox-provider agent runs. Missing/falsy raises :class:`ProviderError`.
* ``context["worktree"]`` (optional, str) — absolute path of the box's
  worktree. When set, it is forwarded as ``cb exec --workdir <worktree>
  <box> ...``. Confirmed at M1b: the real ``cb exec`` CLI does accept a
  ``--workdir <path>`` flag; no adjustment needed.

**M1b resolution — how ``box``/``worktree`` actually get into ``context``.**
Conductor's ``WorkflowContext.build_for_agent()`` (the only thing that
builds the ``context`` dict this provider receives) never places a bare
top-level key into that dict for anything other than ``workflow``/``context``
metadata and per-step ``{"output": ...}``-wrapped entries — there is no
declarative (YAML ``script``/``set`` step) way to land a literal
``context["box"]``. Two supported mechanisms, in precedence order:

1. **Workflow input (CLI-compatible, no embedding required).** Declare
   ``workflow.input.box`` / ``workflow.input.worktree`` in the YAML and pass
   them at launch: ``conductor run workflow.yaml --input box=<id> --input
   worktree=<path>``. Every accumulate-mode ``context`` dict carries the full
   ``workflow.input`` unconditionally (see ``_LOCAL_RENDER_AGENT_TYPES`` /
   the non-explicit branch in ``context.py``), so this provider falls back
   to ``context["workflow"]["input"]["box"]`` /
   ``context["workflow"]["input"]["worktree"]`` when the flat top-level key
   is absent. This is the mechanism used by ``conductor run`` end to end and
   by the module's shipped workflow templates.
2. **Direct context key (embedding only).** A caller embedding
   ``WorkflowEngine`` directly in Python (not going through the ``conductor
   run`` CLI) can still set ``context["box"]``/``context["worktree"]``
   literally by wrapping/subclassing this provider's ``execute()`` and
   injecting the keys before delegating — useful when the box id is only
   known at Python call time and a workflow-input round-trip is undesired.
   This path remains supported (checked first) but is not exercised by the
   CLI-driven module templates.

Two more values used by the invocation are *not* context keys:

* ``<role>`` (the ``--agent`` value) is the step's own ``agent.name`` —
  Conductor's ``AgentDef.name``, not a context lookup.
* The structured-output schema is the step's own ``agent.output`` field
  (Conductor's existing ``output:`` YAML block) — also not a context key.
  When declared, its JSON Schema is injected into the prompt (this
  provider's ``structured_output`` capability is ``"prompt_injection"``,
  not native), and the terminal result text is parsed as JSON with bounded
  recovery: on a parse failure, a short follow-up prompt is resent via
  ``claude --resume <session_id>`` (the session id from the ``stream-json``
  ``system``/``init`` event), up to
  ``agent.retry.max_parse_recovery_attempts`` (default 2) times.

Not yet implemented / left for the live-box pass (M1b):

* Non-root user enforcement: this provider assumes ``cb exec`` already runs
  as claudebox's non-root ``agent`` user by convention (bypassPermissions is
  refused as root). No explicit ``--user`` flag is passed. If the live CLI
  defaults to root, add ``--user agent`` in :meth:`_build_argv`.
* Workflow ``tools:`` allowlists are refused loudly (not silently dropped) —
  tool access inside the box is governed by the box's own claude
  configuration/persona, not Conductor's per-agent ``tools:`` field.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final

from conductor.exceptions import ProviderError, ValidationError
from conductor.providers.base import AgentOutput, AgentProvider, EventCallback
from conductor.providers.capabilities import ProviderCapabilities

if TYPE_CHECKING:
    from conductor.config.schema import AgentDef, OutputField

logger = logging.getLogger(__name__)

# Default SDK-recognized model alias when neither the agent nor the workflow
# sets one. `claude -p --model` accepts short aliases (opus|sonnet|haiku) or
# full dated ids.
_DEFAULT_MODEL: Final[str] = "sonnet"

# Fallback `cb` binary name, resolved via PATH. Overridable per-instance
# (constructor `cb_binary` kwarg) or globally via
# CONDUCTOR_CLAUDEBOX_CB_PATH — the latter is how the hermetic test suite
# points this provider at a fake `cb` script without touching PATH.
_DEFAULT_CB_BINARY: Final[str] = "cb"
_CB_PATH_ENV_VAR: Final[str] = "CONDUCTOR_CLAUDEBOX_CB_PATH"

# Bounded parse-recovery attempts when the agent declares an `output:`
# schema but the model's final text doesn't parse as JSON. Mirrors
# providers/claude.py's RetryConfig.max_parse_recovery_attempts default.
_DEFAULT_PARSE_RECOVERY_ATTEMPTS: Final[int] = 2

# Preview length for tool_result content forwarded in agent_tool_complete
# events (mirrors claude_agent_sdk.py's _TOOL_RESULT_PREVIEW_LEN).
_TOOL_RESULT_PREVIEW_LEN: Final[int] = 500

# Bound on how many stray non-JSON stdout lines (CLI banners, log noise) we
# retain per run so ProviderError can surface a tail of them on a non-zero
# exit without holding unbounded stdout in memory.
_NOISE_LINE_CAP: Final[int] = 20

# StreamReader buffer limit for the streaming `cb exec ... claude ...`
# subprocess. asyncio's default is 64 KiB, but `claude --output-format
# stream-json` emits one JSON object per line, and a single large
# `tool_result` (big Bash/test-runner output) or file-write body easily
# exceeds that — `readline()` then raises `ValueError: Separator is found,
# but chunk is longer than limit` and kills the workflow. 64 MiB covers
# realistic outputs while still bounding memory; a pathological line beyond
# it still raises (known bound).
_STREAM_READ_LIMIT: Final[int] = 64 * 1024 * 1024

# How long to wait for the subprocess to exit after terminate() before
# escalating to kill() — used both for interrupt and for max_session_seconds
# timeout cleanup.
_TERMINATE_GRACE_SECONDS: Final[float] = 5.0

_FENCE_RE: Final[re.Pattern[str]] = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

_PARSE_RECOVERY_PROMPT: Final[str] = (
    "Your previous response could not be parsed as JSON matching the requested "
    "schema. Respond again with ONLY a single valid JSON object matching that "
    "schema -- no prose, no markdown code fence, nothing else."
)


def _build_field_schema(field_def: OutputField, depth: int = 0) -> dict[str, Any]:
    """Translate a single ``OutputField`` into a JSON-Schema fragment.

    Self-contained (not shared with claude_agent_sdk.py's equivalent
    helper): this provider injects the schema as prompt *text*, not as an
    SDK-native ``output_format`` option, so it has its own small builder
    rather than depending on a sibling provider module's private API.
    """
    if depth > 10:
        raise ProviderError("Output schema nesting exceeds 10 levels", is_retryable=False)

    schema: dict[str, Any] = {"type": field_def.type}
    if field_def.description:
        schema["description"] = field_def.description
    if field_def.type == "object" and field_def.properties:
        schema["properties"] = {
            name: _build_field_schema(f, depth + 1) for name, f in field_def.properties.items()
        }
        schema["required"] = list(field_def.properties.keys())
    if field_def.type == "array" and field_def.items:
        schema["items"] = _build_field_schema(field_def.items, depth + 1)
    return schema


def _output_schema_instructions(output: dict[str, OutputField]) -> str:
    """Build the prompt-injection instruction block for a declared ``output:`` schema."""
    schema = {
        "type": "object",
        "properties": {name: _build_field_schema(f) for name, f in output.items()},
        "required": list(output.keys()),
    }
    return (
        "Respond with a single JSON object matching this JSON Schema, and "
        "nothing else (no prose, no markdown code fence):\n" + json.dumps(schema, indent=2)
    )


def _try_parse_json(text: str) -> tuple[dict[str, Any], bool]:
    """Best-effort JSON object extraction from a model's final text.

    Returns ``(content, ok)``. ``ok`` is True only when a JSON *object* was
    successfully parsed (bare arrays/scalars don't satisfy an ``output:``
    schema contract). On failure, ``content`` is a ``{"response": text}``
    fallback wrapper so callers always get a dict back.

    Recovery order: bare ``json.loads`` -> a fenced ```json ... ``` block ->
    the outermost ``{...}`` substring. This is a *local, same-text* recovery
    heuristic; the caller (:meth:`ClaudeboxProvider.execute`) layers a
    session-level recovery (re-asking the model) on top when this fails.
    """
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed, True
    except json.JSONDecodeError:
        pass

    fenced = _FENCE_RE.search(stripped)
    if fenced:
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed, True

    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        with contextlib.suppress(json.JSONDecodeError):
            parsed = json.loads(stripped[start : end + 1])
            if isinstance(parsed, dict):
                return parsed, True

    return {"response": text}, False


def _classify_retryable(text: str, exit_code: int) -> bool:
    """Heuristically classify a claude-subprocess failure as retryable.

    We only have exit code + stderr/result text to work with (no typed SDK
    exceptions, unlike claude_agent_sdk.py) so classification is
    string-based, mirroring claude_agent_sdk.py's `_is_retryable_result`
    keyword heuristics.
    """
    del exit_code  # kept for signature symmetry / future use
    t = text.lower()
    # OAuth/session-expiry failures first, ahead of the retryable keyword
    # groups below: a message like "OAuth session expired ... connection"
    # must not be classified retryable just because "connection" also
    # appears in it (kentra-io/harness#3 — a dead box's expired session
    # churned through 3 retry layers for ~52 minutes before this fix).
    if any(k in t for k in ("oauth", "session expired", "could not be refreshed")):
        return False
    if any(k in t for k in ("unauthorized", "401", "403", "invalid api key", "authentication")):
        return False
    if any(k in t for k in ("429", "rate limit", "quota", "overloaded")):
        return True
    if any(k in t for k in ("500", "502", "503", "504", "internal server error")):
        return True
    return bool(any(k in t for k in ("network", "connection", "econnreset", "timed out")))


def _safe_callback(callback: EventCallback, event_type: str, data: dict[str, Any]) -> None:
    try:
        callback(event_type, data)
    except Exception:
        logger.debug("Error in event_callback for %s", event_type, exc_info=True)


@dataclass
class _RunOutcome:
    """Accumulated state from streaming one ``claude -p ... --output-format stream-json`` run."""

    content_parts: list[str] = field(default_factory=list)
    result_text: str | None = None
    result_is_error: bool = False
    result_error_message: str | None = None
    session_id: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    saw_terminal_result: bool = False
    partial: bool = False
    turn_count: int = 0
    pending_tools: dict[str, str] = field(default_factory=dict)
    noise_lines: list[str] = field(default_factory=list)


def _record_noise_line(outcome: _RunOutcome, text: str) -> None:
    """Retain a bounded tail of stray non-JSON stdout lines on ``outcome``.

    Keeps at most ``_NOISE_LINE_CAP`` lines (dropping the oldest) so a long
    run's incidental CLI banner/log noise doesn't grow unbounded, while
    still leaving a diagnostic tail available if the subprocess exits
    non-zero (see the non-zero-exit ``ProviderError`` in ``_run_once``).
    """
    outcome.noise_lines.append(text)
    if len(outcome.noise_lines) > _NOISE_LINE_CAP:
        del outcome.noise_lines[: len(outcome.noise_lines) - _NOISE_LINE_CAP]


def _diagnostic_tails(outcome: _RunOutcome) -> tuple[str, str]:
    """Bounded stdout-noise and agent-content tails for the non-zero-exit path.

    ``_nonzero_exit_detail`` uses both halves; ``_run_once``'s classification
    uses only the content half (it re-derives its own full-history stdout
    noise string separately -- see the comment at its call site). Kept as
    one source of truth so both bound the same accumulated ``outcome``.
    """
    stdout_tail = " | ".join(line.strip() for line in outcome.noise_lines[-5:] if line.strip())
    content_tail = " | ".join(p.strip() for p in outcome.content_parts[-3:] if p.strip())[-500:]
    return stdout_tail, content_tail


def _nonzero_exit_detail(stderr_text: str, outcome: _RunOutcome) -> str:
    """Best-effort diagnostic for a non-zero ``claude`` subprocess exit.

    Falls back through stderr, then a bounded tail of stray stdout noise
    lines, then a tail of agent-message text content: a dead box's OAuth
    session expiry surfaces only there (empty stderr, no stdout noise, the
    failure text is the agent's last message — kentra-io/harness#3).
    """
    stdout_tail, content_tail = _diagnostic_tails(outcome)
    return (
        stderr_text.strip()
        or stdout_tail
        or content_tail
        or "(no stderr, stdout, or agent-content diagnostics)"
    )


def _process_line(
    outcome: _RunOutcome,
    raw_line: bytes,
    event_callback: EventCallback | None,
) -> None:
    """Parse one ``stream-json`` line and fold it into ``outcome`` / emit events.

    Recognizes the standard ``claude -p --output-format stream-json``
    envelope: ``system`` (subtype ``init`` carries ``session_id``),
    ``assistant`` (an Anthropic Messages-API-shaped ``message`` with
    ``content`` blocks + ``usage``), ``user`` (tool_result blocks), and the
    terminal ``result`` (``is_error``, ``result`` text, cumulative
    ``usage``, ``total_cost_usd``, ``session_id``). Unrecognized line
    shapes and non-JSON lines are skipped (forward-compatible, and
    tolerant of stray CLI banner/log noise on stdout).
    """
    text = raw_line.decode("utf-8", errors="replace").strip()
    if not text:
        return
    try:
        event = json.loads(text)
    except json.JSONDecodeError:
        logger.debug("Skipping non-JSON stream-json line: %r", text[:200])
        _record_noise_line(outcome, text)
        return
    if not isinstance(event, dict):
        _record_noise_line(outcome, text)
        return

    event_type = event.get("type")

    if event_type == "system":
        if event.get("subtype") == "init":
            outcome.session_id = event.get("session_id") or outcome.session_id
            outcome.model = event.get("model") or outcome.model
        return

    if event_type == "assistant":
        message = event.get("message") or {}
        outcome.turn_count += 1
        if event_callback:
            _safe_callback(event_callback, "agent_turn_start", {"turn": outcome.turn_count})

        if message.get("model"):
            outcome.model = message["model"]

        usage = message.get("usage") or {}
        if usage:
            outcome.input_tokens += usage.get("input_tokens", 0) or 0
            outcome.output_tokens += usage.get("output_tokens", 0) or 0
            outcome.cache_read_tokens += usage.get("cache_read_input_tokens", 0) or 0
            outcome.cache_write_tokens += usage.get("cache_creation_input_tokens", 0) or 0

        blocks = message.get("content") or []
        has_tool_use = any(isinstance(b, dict) and b.get("type") == "tool_use" for b in blocks)
        for block in blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_content = block.get("text", "")
                if text_content:
                    outcome.content_parts.append(text_content)
                    if event_callback:
                        _safe_callback(event_callback, "agent_message", {"content": text_content})
            elif btype == "thinking":
                thinking = block.get("thinking", "")
                if thinking and event_callback:
                    _safe_callback(event_callback, "agent_reasoning", {"content": thinking})
            elif btype == "tool_use":
                tool_name = block.get("name", "unknown")
                tool_id = block.get("id", "")
                outcome.pending_tools[tool_id] = tool_name
                if event_callback:
                    _safe_callback(
                        event_callback,
                        "agent_tool_start",
                        {"tool_name": tool_name, "arguments": block.get("input")},
                    )
        if has_tool_use and event_callback:
            _safe_callback(event_callback, "agent_turn_start", {"turn": "awaiting_model"})
        return

    if event_type == "user":
        message = event.get("message") or {}
        for block in message.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id", "")
            tool_name = outcome.pending_tools.pop(tool_use_id, "unknown")
            content = block.get("content", "")
            result_str = str(content)[:_TOOL_RESULT_PREVIEW_LEN] if content else None
            if event_callback:
                _safe_callback(
                    event_callback,
                    "agent_tool_complete",
                    {"tool_name": tool_name, "result": result_str},
                )
        return

    if event_type == "result":
        outcome.saw_terminal_result = True
        outcome.result_text = event.get("result")
        outcome.result_is_error = bool(event.get("is_error"))
        outcome.result_error_message = str(event.get("result")) if outcome.result_is_error else None
        outcome.session_id = event.get("session_id") or outcome.session_id
        usage = event.get("usage") or {}
        if usage:
            # The terminal result's usage is the CUMULATIVE session total —
            # replace rather than add (mirrors claude_agent_sdk.py).
            outcome.input_tokens = usage.get("input_tokens", outcome.input_tokens) or 0
            outcome.output_tokens = usage.get("output_tokens", outcome.output_tokens) or 0
            outcome.cache_read_tokens = (
                usage.get("cache_read_input_tokens", outcome.cache_read_tokens) or 0
            )
            outcome.cache_write_tokens = (
                usage.get("cache_creation_input_tokens", outcome.cache_write_tokens) or 0
            )
        cost = event.get("total_cost_usd")
        if cost is not None:
            outcome.cost_usd = cost
        return
    # Unknown/forward-compatible event type (e.g. system/api_retry): no-op.


class ClaudeboxProvider(AgentProvider):
    """Drives a ``claude`` CLI subprocess inside a claudebox sandbox via ``cb exec``.

    See the module docstring for the workflow-``context`` key contract
    (``context["box"]`` / ``context["worktree"]``) and the structured-output
    prompt-injection + session-resume recovery design.
    """

    CAPABILITIES = ProviderCapabilities(
        tier="experimental",
        # The box's own claude configuration (persona/skills) governs tool
        # access; Conductor's `runtime.mcp_servers` / per-agent `tools:` are
        # not forwarded. A non-empty `tools:` allowlist is refused loudly
        # at execute() time rather than silently dropped.
        mcp_tools=False,
        workflow_tools_passthrough=False,
        # stream-json lines are translated into agent_message/agent_tool_*
        # events as they arrive.
        streaming_events=True,
        # `thinking` content blocks are forwarded as agent_reasoning.
        agent_reasoning_events=True,
        # No `reasoning.effort` -> CLI flag plumbing exists (the fixed
        # invocation has no thinking-budget flag).
        reasoning_effort=None,
        # Schema instructions are appended to the prompt and the terminal
        # result text is JSON-parsed with bounded (same-text + session
        # --resume) recovery -- not SDK-native enforcement.
        structured_output="prompt_injection",
        # interrupt_signal is raced against stdout reads; on fire the
        # subprocess is terminated and partial output is returned.
        interrupt=True,
        # Each subprocess call is wrapped in `asyncio.wait_for` using the
        # resolved agent/provider max_session_seconds (falling back to the
        # provider's `timeout`).
        max_session_seconds=True,
        # Each `claude -p` invocation is a fresh CLI session; Conductor does
        # not persist/replay `--resume` session ids across `conductor resume`.
        checkpoint_resume=False,
        # input/output/cache tokens + model come from the terminal `result`
        # event's cumulative `usage` block.
        usage_tracking=True,
        # No shared mutable state across calls -- safe to run N in parallel
        # (each spawns its own subprocess against, presumably, its own box).
        concurrent_safe=True,
        upstream_pin=None,
        maintainer="kentra (M1)",
    )

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_agent_iterations: int | None = None,
        max_session_seconds: float | None = None,
        auth_token: str | None = None,
        base_url: str | None = None,
        cb_binary: str | None = None,
    ) -> None:
        """Initialize the claudebox provider.

        Args:
            model: Default model alias/id (``opus``/``sonnet``/``haiku`` or a
                full dated id). Defaults to ``"sonnet"``.
            temperature: Accepted for constructor-shape parity with the
                other providers; unused -- `claude -p` has no per-call
                temperature flag in the fixed invocation.
            max_tokens: Accepted for parity; unused -- no per-call
                max-output-tokens flag exists for `claude -p`.
            timeout: Fallback wall-clock bound (seconds) applied when
                neither the agent nor the provider sets
                `max_session_seconds`. Mirrors `RuntimeConfig.timeout`.
            max_agent_iterations: Accepted for parity; unused -- the `claude`
                CLI manages its own internal turn loop; no flag exists to cap
                it in the fixed invocation.
            max_session_seconds: Provider-level default wall-clock bound for
                the whole subprocess call. Per-agent
                `agent.max_session_seconds` overrides this.
            auth_token: Reserved Stage-4 gateway slot. When set, injected
                into the subprocess env as `ANTHROPIC_AUTH_TOKEN`. `None` by
                default (unused today).
            base_url: Reserved Stage-4 gateway slot. When set, injected as
                `ANTHROPIC_BASE_URL`. `None` by default (unused today).
            cb_binary: Override for the `cb` executable path/name. Defaults
                to the `CONDUCTOR_CLAUDEBOX_CB_PATH` env var, then the bare
                name `"cb"` (resolved via `PATH`). Exists primarily so tests
                can point this provider at a fake `cb` script.
        """
        self._default_model = model or _DEFAULT_MODEL
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        self._default_timeout = timeout
        self._default_max_agent_iterations = max_agent_iterations
        self._default_max_session_seconds = max_session_seconds
        self._auth_token = auth_token
        self._base_url = base_url
        self._cb_binary = cb_binary or os.environ.get(_CB_PATH_ENV_VAR) or _DEFAULT_CB_BINARY

    # ------------------------------------------------------------------
    # AgentProvider interface
    # ------------------------------------------------------------------

    async def execute(
        self,
        agent: AgentDef,
        context: dict[str, Any],
        rendered_prompt: str,
        tools: list[str] | None = None,
        interrupt_signal: asyncio.Event | None = None,
        event_callback: EventCallback | None = None,
    ) -> AgentOutput:
        self._check_tools(tools, agent)

        box = context.get("box") or self._workflow_input(context).get("box")
        if not box:
            raise ProviderError(
                "ClaudeboxProvider requires a 'box' key in the workflow context "
                "(the claudebox box/container id to `cb exec` into), or a "
                "'box' workflow input. Set it via `conductor run ... --input "
                "box=<id>` (declare `input.box` in the workflow YAML), or via "
                "a `set`/`script` step / embedding-level context injection "
                "before any claudebox-provider agent runs.",
                is_retryable=False,
            )
        worktree = context.get("worktree") or self._workflow_input(context).get("worktree")

        model = agent.model or self._default_model
        timeout = self._resolve_session_timeout(agent)
        env = self._build_env()

        prompt = rendered_prompt
        if agent.output:
            prompt = f"{rendered_prompt}\n\n{_output_schema_instructions(agent.output)}"

        argv = self._build_argv(box, worktree, agent.name, model, prompt)
        outcome = await self._run_once(argv, env, interrupt_signal, event_callback, timeout)

        if outcome.partial:
            return self._build_output(outcome, agent, model, partial=True)

        if not agent.output:
            content = {"response": outcome.result_text or "\n".join(outcome.content_parts)}
            return self._build_output(outcome, agent, model, content=content)

        # Structured output: same-text recovery already attempted inside
        # _try_parse_json; layer session-level recovery (re-ask the model)
        # on top, bounded by max_parse_recovery_attempts.
        combined = outcome.result_text or "\n".join(outcome.content_parts)
        content, ok = _try_parse_json(combined)
        max_recovery = self._resolve_parse_recovery_attempts(agent)
        attempts = 0
        while not ok and attempts < max_recovery and outcome.session_id:
            attempts += 1
            argv = self._build_argv(
                box,
                worktree,
                agent.name,
                model,
                _PARSE_RECOVERY_PROMPT,
                resume_session_id=outcome.session_id,
            )
            outcome = await self._run_once(argv, env, interrupt_signal, event_callback, timeout)
            if outcome.partial:
                return self._build_output(outcome, agent, model, partial=True)
            combined = outcome.result_text or "\n".join(outcome.content_parts)
            content, ok = _try_parse_json(combined)

        if not ok:
            raise ValidationError(
                f"Agent '{agent.name}' declared an output schema but claude did "
                f"not return parseable JSON after {attempts} recovery attempt(s): "
                f"{combined[:200]!r}",
                suggestion=(
                    "Ensure the agent's prompt/system_prompt instructs the model "
                    "to emit a single JSON object matching the declared `output:` "
                    "fields, or remove the `output:` schema."
                ),
            )
        return self._build_output(outcome, agent, model, content=content)

    async def validate_connection(self) -> bool:
        """Check that ``cb`` is on PATH and can enumerate boxes (``cb ls``).

        No specific box is known at provider-construction time (the box id
        arrives per-execution via `context["box"]`), so this is a
        best-effort "is claudebox functional at all" probe, not a
        per-box reachability check.
        """
        if shutil.which(self._cb_binary) is None:
            logger.warning(
                "claudebox CLI %r not found on PATH. Install claudebox and "
                "ensure `cb` is on PATH, or set CONDUCTOR_CLAUDEBOX_CB_PATH.",
                self._cb_binary,
            )
            return False

        try:
            process = await asyncio.create_subprocess_exec(
                self._cb_binary,
                "ls",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            logger.warning("Failed to run `%s ls`: %s", self._cb_binary, exc)
            return False

        try:
            await asyncio.wait_for(process.communicate(), timeout=10)
        except TimeoutError:
            process.kill()
            await process.wait()
            logger.warning("`%s ls` timed out during validate_connection().", self._cb_binary)
            return False

        if process.returncode != 0:
            logger.warning(
                "`%s ls` exited with code %s during validate_connection().",
                self._cb_binary,
                process.returncode,
            )
        return process.returncode == 0

    async def close(self) -> None:
        """No-op -- this provider holds no persistent resources across calls."""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _workflow_input(context: dict[str, Any]) -> dict[str, Any]:
        """Return ``context["workflow"]["input"]``, tolerating either being absent.

        See the module docstring's "M1b resolution" section: this is the
        fallback lookup path for ``box``/``worktree`` when the workflow was
        launched via ``conductor run ... --input box=<id> --input
        worktree=<path>`` rather than an embedding caller setting the flat
        top-level context keys directly.
        """
        workflow = context.get("workflow")
        if not isinstance(workflow, dict):
            return {}
        workflow_input = workflow.get("input")
        return workflow_input if isinstance(workflow_input, dict) else {}

    @staticmethod
    def _check_tools(tools: list[str] | None, agent: AgentDef) -> None:
        """Refuse a non-empty workflow `tools:` allowlist rather than silently drop it.

        Tool access inside the box is governed by the box's own claude
        configuration (persona/skills materialized by the agent-definition
        primitive), not by Conductor's per-agent `tools:` field -- there is
        no translation from workflow tool names to that configuration.
        """
        if tools:
            raise ProviderError(
                f"Agent '{agent.name}' resolves to tools={tools!r} (declared on "
                "the agent or inherited from the workflow-level 'tools:' list), "
                "but claudebox does not support workflow tools allowlists -- "
                "tool access inside the box is governed by the box's own "
                "claude configuration, not Conductor's `tools:` field.",
                suggestion=(
                    "Omit `tools:` for claudebox-provider agents; configure "
                    "tool access via the box's persona/skills instead."
                ),
                is_retryable=False,
            )

    def _resolve_session_timeout(self, agent: AgentDef) -> float | None:
        """Resolve the wall-clock bound for one subprocess call.

        Precedence: per-agent `max_session_seconds` -> provider-level
        `max_session_seconds` -> provider-level `timeout` (the
        `runtime.timeout` fallback) -> `None` (unbounded).
        """
        if agent.max_session_seconds is not None:
            return agent.max_session_seconds
        if self._default_max_session_seconds is not None:
            return self._default_max_session_seconds
        return self._default_timeout

    def _resolve_parse_recovery_attempts(self, agent: AgentDef) -> int:
        retry = getattr(agent, "retry", None)
        value = getattr(retry, "max_parse_recovery_attempts", None) if retry is not None else None
        return value if value is not None else _DEFAULT_PARSE_RECOVERY_ATTEMPTS

    def _build_env(self) -> dict[str, str] | None:
        """Build the subprocess env, injecting the reserved gateway slot if configured.

        Returns `None` (inherit the parent's environment unchanged) unless
        `auth_token`/`base_url` were explicitly configured -- today's M1
        default -- or `CLAUDE_CODE_LONG_LIVED_TOKEN` is set in the parent
        env. When any apply, they're layered on top of a copy of the
        current environment (never mutating `os.environ` itself).

        `CLAUDE_CODE_LONG_LIVED_TOKEN` (a 1-year non-rotating `claude
        setup-token` credential, held by the orchestration daemon and read
        from the macOS keychain) is remapped to `CLAUDE_CODE_OAUTH_TOKEN` --
        the name `claude` itself reads -- so agent boxes carry no
        credentials file at all and never hit the OAuth refresh-rotation
        race (kentra-io/harness#3).
        """
        long_lived = os.environ.get("CLAUDE_CODE_LONG_LIVED_TOKEN")
        if self._auth_token is None and self._base_url is None and not long_lived:
            return None
        env = dict(os.environ)
        if long_lived:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = long_lived
        if self._base_url:
            env["ANTHROPIC_BASE_URL"] = self._base_url
        if self._auth_token:
            env["ANTHROPIC_AUTH_TOKEN"] = self._auth_token
        return env

    def _build_argv(
        self,
        box: str,
        worktree: str | None,
        role: str,
        model: str,
        prompt: str,
        resume_session_id: str | None = None,
    ) -> list[str]:
        """Build the `cb exec ... claude ...` argument vector."""
        argv = [self._cb_binary, "exec"]
        if os.environ.get("CLAUDE_CODE_LONG_LIVED_TOKEN"):
            # Bare name: docker exec forwards the value from the client env,
            # which _build_env populated -- the secret never enters argv
            # (kentra-io/harness#3: env-auth boxes carry no session file).
            argv += ["-e", "CLAUDE_CODE_OAUTH_TOKEN"]
        if worktree:
            argv += ["--workdir", str(worktree)]
        argv.append(str(box))
        argv += ["claude", "-p", prompt, "--agent", role, "--model", model]
        if resume_session_id:
            argv += ["--resume", resume_session_id]
        argv += [
            "--permission-mode",
            "bypassPermissions",
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        return argv

    async def _run_once(
        self,
        argv: list[str],
        env: dict[str, str] | None,
        interrupt_signal: asyncio.Event | None,
        event_callback: EventCallback | None,
        timeout: float | None,
    ) -> _RunOutcome:
        """Spawn one `cb exec ... claude ...` subprocess and stream its output."""
        if event_callback:
            _safe_callback(event_callback, "agent_turn_start", {"turn": "awaiting_model"})

        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                limit=_STREAM_READ_LIMIT,
            )
        except FileNotFoundError as exc:
            raise ProviderError(
                f"claudebox CLI not found: {argv[0]!r} is not on PATH",
                suggestion=(
                    f"Install claudebox and ensure `cb` is on PATH, or set {_CB_PATH_ENV_VAR}."
                ),
                is_retryable=False,
            ) from exc
        except OSError as exc:
            raise ProviderError(
                f"Failed to start claudebox subprocess: {exc}",
                is_retryable=True,
            ) from exc

        assert process.stderr is not None
        stderr_task: asyncio.Future[bytes] = asyncio.ensure_future(process.stderr.read())

        try:
            if timeout is not None:
                outcome = await asyncio.wait_for(
                    self._read_loop(process, interrupt_signal, event_callback), timeout=timeout
                )
            else:
                outcome = await self._read_loop(process, interrupt_signal, event_callback)
        except TimeoutError:
            await self._terminate(process)
            stderr_task.cancel()
            raise ProviderError(
                f"claudebox agent exceeded max_session_seconds={timeout:.0f}s",
                is_retryable=False,
            ) from None
        except asyncio.CancelledError:
            await self._terminate(process)
            stderr_task.cancel()
            raise

        if outcome.partial:
            await self._terminate(process)
            stderr_task.cancel()
            return outcome

        await process.wait()
        stderr_bytes = b""
        with contextlib.suppress(Exception):
            stderr_bytes = await asyncio.wait_for(stderr_task, timeout=5)
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")

        exit_code = process.returncode
        if exit_code != 0:
            # Classify against every CLI-authored signal, not stderr alone —
            # this mirrors ab0ff4c's original `diag` (stderr + ALL retained
            # noise_lines + result_error_message), widened further to also
            # include the agent-content tail: a dead box's OAuth session
            # expiry surfaces only as agent *message content* (stdout JSON)
            # with stderr/noise/result all empty — kentra-io/harness#3. Note
            # this deliberately reads the FULL noise_lines here, not just the
            # bounded tail `_diagnostic_tails`/`detail` use for the reported
            # message — a transient signal earlier than the last 5 lines must
            # still be classifiable even if it's not worth quoting in full.
            # `_classify_retryable` checks OAuth/session patterns first so
            # this combined text can't be misclassified retryable by an
            # incidental keyword like "connection".
            _, content_tail = _diagnostic_tails(outcome)
            detail = _nonzero_exit_detail(stderr_text, outcome)
            noise_all = " | ".join(line.strip() for line in outcome.noise_lines if line.strip())
            result_error = outcome.result_error_message or ""
            classify_text = f"{stderr_text}\n{noise_all}\n{content_tail}\n{result_error}"
            raise ProviderError(
                f"claude subprocess exited with code {exit_code}: {detail}",
                is_retryable=_classify_retryable(classify_text, exit_code or 0),
            )
        if not outcome.saw_terminal_result:
            raise ProviderError(
                "claude subprocess exited 0 but the stream-json output never "
                "produced a terminal `result` event (malformed/truncated stream)"
                + (f"; stderr: {stderr_text.strip()}" if stderr_text.strip() else ""),
                is_retryable=False,
            )
        if outcome.result_is_error:
            raise ProviderError(
                f"claude reported an error result: "
                f"{outcome.result_error_message or '(no message)'}",
                is_retryable=_classify_retryable(outcome.result_error_message or "", 0),
            )
        return outcome

    @staticmethod
    async def _read_loop(
        process: asyncio.subprocess.Process,
        interrupt_signal: asyncio.Event | None,
        event_callback: EventCallback | None,
    ) -> _RunOutcome:
        """Read `stream-json` lines until EOF or `interrupt_signal` fires."""
        outcome = _RunOutcome()
        assert process.stdout is not None

        while True:
            read_task: asyncio.Future[bytes] = asyncio.ensure_future(process.stdout.readline())
            interrupt_task: asyncio.Future[bool] | None = None
            waiters: set[asyncio.Future[Any]] = {read_task}
            if interrupt_signal is not None:
                interrupt_task = asyncio.ensure_future(interrupt_signal.wait())
                waiters.add(interrupt_task)

            try:
                done, pending = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            except asyncio.CancelledError:
                for t in waiters:
                    t.cancel()
                raise

            for t in pending:
                t.cancel()

            if interrupt_task is not None and interrupt_task in done:
                read_task.cancel()
                outcome.partial = True
                return outcome

            line = read_task.result()
            if not line:
                return outcome

            _process_line(outcome, line, event_callback)

    @staticmethod
    async def _terminate(process: asyncio.subprocess.Process) -> None:
        """Terminate (escalating to kill) a subprocess, tolerating a race with natural exit."""
        if process.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=_TERMINATE_GRACE_SECONDS)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()

    @staticmethod
    def _build_output(
        outcome: _RunOutcome,
        agent: AgentDef,
        model: str,
        content: dict[str, Any] | None = None,
        partial: bool = False,
    ) -> AgentOutput:
        combined = outcome.result_text or "\n".join(outcome.content_parts)
        if content is None:
            content = {"response": combined}
        total = outcome.input_tokens + outcome.output_tokens
        return AgentOutput(
            content=content,
            raw_response={
                "result": combined,
                "session_id": outcome.session_id,
                # The claude CLI's self-reported total_cost_usd is preserved
                # here for debugging/audit parity with the CLI's own
                # accounting. AgentOutput has no `cost` field -- Conductor's
                # engine computes the authoritative cost from
                # (model, input_tokens, output_tokens) via its own pricing
                # table/hook chain, so this value is not double-counted.
                "total_cost_usd": outcome.cost_usd,
            },
            tokens_used=total or None,
            input_tokens=outcome.input_tokens or None,
            output_tokens=outcome.output_tokens or None,
            cache_read_tokens=outcome.cache_read_tokens or None,
            cache_write_tokens=outcome.cache_write_tokens or None,
            model=outcome.model or model,
            partial=partial,
        )
