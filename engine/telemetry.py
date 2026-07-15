"""Engine-neutral observability contracts.

The inference engine emits a small set of lifecycle measurements through this
interface without importing Prometheus, OpenTelemetry, or a logging backend.
Library and benchmark users get the no-op implementation by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from engine.scheduler import Request


@dataclass(frozen=True)
class SchedulerSnapshot:
    """Low-cardinality state captured once after each scheduler step."""

    waiting_requests: int
    running_requests: int
    free_blocks: int
    used_blocks: int
    decode_batch_size: int
    admitted_requests: int
    generated_tokens: int
    step_duration_seconds: float


class TelemetrySink(Protocol):
    """Events emitted by the serving path.

    Implementations must keep these methods non-blocking: most calls happen on
    the single engine worker thread and therefore sit on the inference path.
    """

    def request_submitted(self, request: "Request") -> None: ...

    def request_admitted(self, request: "Request", queue_seconds: float) -> None: ...

    def request_first_token(self, request: "Request", ttft_seconds: float) -> None: ...

    def inter_token_latency(self, seconds: float) -> None: ...

    def tokens_generated(self, count: int) -> None: ...

    def request_preempted(self, request: "Request") -> None: ...

    def request_finished(
        self, request: "Request", duration_seconds: float, finish_reason: str
    ) -> None: ...

    def request_aborted(self, request: "Request", duration_seconds: float) -> None: ...

    def prefill_completed(self, token_count: int, duration_seconds: float) -> None: ...

    def decode_completed(self, batch_size: int, duration_seconds: float) -> None: ...

    def scheduler_initialized(self, total_blocks: int) -> None: ...

    def scheduler_step(self, snapshot: SchedulerSnapshot) -> None: ...


class NullTelemetry:
    """Default sink exposed when the scheduler's fast telemetry flag is off."""

    def request_submitted(self, request: "Request") -> None:
        pass

    def request_admitted(self, request: "Request", queue_seconds: float) -> None:
        pass

    def request_first_token(self, request: "Request", ttft_seconds: float) -> None:
        pass

    def inter_token_latency(self, seconds: float) -> None:
        pass

    def tokens_generated(self, count: int) -> None:
        pass

    def request_preempted(self, request: "Request") -> None:
        pass

    def request_finished(
        self, request: "Request", duration_seconds: float, finish_reason: str
    ) -> None:
        pass

    def request_aborted(self, request: "Request", duration_seconds: float) -> None:
        pass

    def prefill_completed(self, token_count: int, duration_seconds: float) -> None:
        pass

    def decode_completed(self, batch_size: int, duration_seconds: float) -> None:
        pass

    def scheduler_initialized(self, total_blocks: int) -> None:
        pass

    def scheduler_step(self, snapshot: SchedulerSnapshot) -> None:
        pass
