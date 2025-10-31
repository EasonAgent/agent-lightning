from typing import Any, Dict, List, Optional, Sequence, Union, Literal
from pydantic import BaseModel, ConfigDict

# ===================================================================================================================
# agentlightning/types/core.py
RolloutStatus = Literal[
    "queuing",  # initial status
    "preparing",  # after the trace is claimed
    "running",  # after receiving the first trace
    "failed",  # crashed
    "succeeded",  # status OK
    "cancelled",  # cancelled by user (or watchdog)
    "requeuing",  # retrying
]
"""The status of a rollout."""
AttemptStatus = Literal[
    # A status is essentially a process.
    # It should not have scheduling/management statuses like "queuing" or "cancelled".
    "preparing",
    "running",
    "failed",
    "succeeded",
    "unresponsive",  # the worker has not reported results for a while
    "timeout",  # the worker has been emitting new logs, but have been working on the task for too long
]
"""The status of an attempt."""

class Attempt(BaseModel):
    """Execution attempt for a rollout, including metadata for retries."""
    rollout_id: str
    """The rollout which this attempt belongs to."""
    attempt_id: str
    """The universal id for current attempt."""
    sequence_id: int
    """The sequence number of the attempt, starting from 1."""
    start_time: float
    """The time when the attempt has started."""
    end_time: Optional[float] = None
    """The time when the attempt has ended."""
    status: AttemptStatus = "preparing"
    """The status of the attempt."""
    worker_id: Optional[str] = None
    """The rollout worker which is executing this attempt."""


class Rollout(BaseModel):
    rollout_id: str
    """Unique identifier for the rollout."""
    input: TaskInput
    """Task input used to generate the rollout."""
    start_time: float
    """Timestamp when the rollout started."""
    end_time: Optional[float] = None
    """Timestamp when the rollout ended."""
    mode: Optional[RolloutMode] = None
    """Execution mode such as `"train"`, `"val"` or `"test"`. See [`RolloutMode`][agentlightning.RolloutMode]."""
    resources_id: Optional[str] = None
    """Identifier of the resources required to execute the rollout."""
    status: RolloutStatus = "queuing"
    """Latest status emitted by the controller."""
    config: RolloutConfig = Field(default_factory=RolloutConfig)
    """Retry and timeout configuration associated with the rollout."""

class AttemptedRollout(Rollout):
    """Rollout paired with the currently active attempt."""
    attempt: Attempt
    """The attempt that is currently processing the rollout."""


class ParallelWorkerBase:
    """Base class for workloads executed across multiple worker processes.

    The lifecycle is orchestrated by the main process:

    * [`init()`][agentlightning.ParallelWorkerBase.init] prepares shared state.
    * Each worker calls [`init_worker()`][agentlightning.ParallelWorkerBase.init_worker] during start-up.
    * [`run()`][agentlightning.ParallelWorkerBase.run] performs the parallel workload.
    * Workers call [`teardown_worker()`][agentlightning.ParallelWorkerBase.teardown_worker] before exiting.
    * The main process finalizes through [`teardown()`][agentlightning.ParallelWorkerBase.teardown].

    Subclasses must implement [`run()`][agentlightning.ParallelWorkerBase.run]
    and can override other lifecycle hooks.
    """

# ===================================================================================================================
# agentlightning/types/tracer.py
class Span(BaseModel):
    """Agent Lightning's canonical span model used for persistence and analytics.

    The model captures the most relevant fields from
    `opentelemetry.sdk.trace.ReadableSpan` instances while preserving unmodeled
    attributes in Pydantic `BaseModel`'s extra storage. This keeps the serialized format
    stable even as upstream OpenTelemetry types evolve.
    """

    model_config = ConfigDict(extra="allow")

    rollout_id: str
    """The rollout which this span belongs to."""
    attempt_id: str
    """The attempt which this span belongs to."""
    sequence_id: int
    """The ID to make spans ordered within a single attempt."""

    # Current ID (in hex, formatted via trace_api.format_*)
    trace_id: str  # one rollout can have traces coming from multiple places
    """The trace ID of the span. One rollout/attempt can have multiple traces.
    This ID comes from the OpenTelemetry trace ID generator.
    """
    span_id: str
    """The span ID of the span. This ID comes from the OpenTelemetry span ID generator."""
    parent_id: Optional[str]
    """The parent span ID of the span."""

    # Core ReadableSpan fields
    name: str
    """The name of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""
    status: TraceStatus
    """The status of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""
    attributes: Attributes
    """The attributes of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""
    events: List[Event]
    """The events of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""
    links: List[Link]
    """The links of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""

    # Timestamps
    start_time: Optional[float]
    """The start time of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""
    end_time: Optional[float]
    """The end time of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""

    # Other parsable fields
    context: Optional[SpanContext]
    """The context of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""
    parent: Optional[SpanContext]
    """The parent context of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""
    resource: OtelResource
    """The resource of the span. See [OpenTelemetry docs](https://opentelemetry.io/docs/concepts/signals/traces/)."""
