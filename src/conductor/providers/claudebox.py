"""Claudebox provider stub (registration skeleton).

This module registers the ``"claudebox"`` provider type end-to-end
(factory, registry, config schema) so it is a valid, instantiable
provider ahead of the real implementation.

The real execution path — driving a claudebox-sandboxed Claude Code
subprocess — lands in M1. Until then, :meth:`ClaudeboxProvider.execute`
raises ``NotImplementedError`` unconditionally; this is intentional and
is covered by the M0 acceptance check (construct via the factory,
never call execute).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from conductor.providers.base import AgentOutput, AgentProvider, EventCallback
from conductor.providers.capabilities import ProviderCapabilities

if TYPE_CHECKING:
    from conductor.config.schema import AgentDef


class ClaudeboxProvider(AgentProvider):
    """Stub provider for running agents inside a claudebox sandbox.

    M0 scope: registration only (factory/registry/schema Literals +
    this class, instantiable and passing ``validate_connection()``).
    The subprocess-driving ``execute()`` implementation lands in M1.
    """

    CAPABILITIES = ProviderCapabilities(
        tier="experimental",
        mcp_tools=False,
        workflow_tools_passthrough=False,
        streaming_events=False,
        agent_reasoning_events=False,
        reasoning_effort=None,
        structured_output="none",
        interrupt=False,
        max_session_seconds=False,
        checkpoint_resume=False,
        usage_tracking=False,
        concurrent_safe=True,
        upstream_pin=None,
        maintainer="kentra (M1 pending)",
    )

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_agent_iterations: int | None = None,
        max_session_seconds: float | None = None,
    ) -> None:
        """Initialize the stub provider.

        Args mirror the other providers' constructor shape so the
        factory can forward the same workflow-level runtime knobs
        once M1 wires them into an actual claudebox invocation. None
        of them are used yet.
        """
        self._default_model = model
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        self._default_timeout = timeout
        self._default_max_agent_iterations = max_agent_iterations
        self._default_max_session_seconds = max_session_seconds

    async def execute(
        self,
        agent: AgentDef,
        context: dict[str, Any],
        rendered_prompt: str,
        tools: list[str] | None = None,
        interrupt_signal: asyncio.Event | None = None,
        event_callback: EventCallback | None = None,
    ) -> AgentOutput:
        """Not implemented until M1.

        Raises:
            NotImplementedError: Always — the subprocess-driving
                implementation lands in M1.
        """
        raise NotImplementedError("ClaudeboxProvider.execute lands in M1")

    async def validate_connection(self) -> bool:
        """Trivially valid — no backend to reach yet.

        Returns:
            Always ``True``. Real connectivity checks (e.g. that a
            claudebox container/image is reachable) land with the M1
            execution implementation.
        """
        return True

    async def close(self) -> None:
        """No-op — the stub holds no resources to release."""
