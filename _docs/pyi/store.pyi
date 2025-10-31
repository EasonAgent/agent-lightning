from typing import Any, Dict, List, Literal, Optional, Sequence

from .types import *



# ===================================================================================================================
# agentlightning/store/base.py
""" https://microsoft.github.io/agent-lightning/stable/deep-dive/store/
LightningStore: 数据存储抽象类

"""
class _UnsetType:
    """A sentinel type to indicate an unset value."""
UNSET = _UnsetType()
Unset = _UnsetType  # Alias for convenience


class LightningStore:
    """Contract for the persistent control-plane that coordinates training rollouts.

    A `LightningStore` mediates every interaction between algorithms and runners:

    - **Rollout lifecycle:** accept new rollouts, queue them for execution, create attempts,
      and drive the rollout status machine (`"queuing"` → `"preparing"` → `"running"` →
      `{"succeeded","failed","cancelled"}` or `"requeuing"` when a retry is justified).
    - **Attempt tracking:** record each execution attempt, including progress heartbeats,
      retry sequencing, and terminal states such as `"timeout"` or `"unresponsive"`.
    - **Span ingest:** capture structured telemetry emitted by runners (either as native
      [`Span`][agentlightning.Span] objects or as `opentelemetry.sdk.trace.ReadableSpan`
      instances) so that algorithms can reconstruct trajectories and rewards.
    - **Resource versioning:** manage immutable snapshots of named resources
      (prompt templates, model checkpoints, proxy endpoints, …) and expose a single
      "latest" snapshot that runners can fetch just after claiming work.

    Implementations must provide thread-safe/async-safe semantics: each coroutine should
    appear atomic to callers even when multiple algorithms or runners call the API concurrently.
    Unless stated otherwise, missing identifiers should result in a `ValueError`.
    """

    async def enqueue_rollout(
        self,
        input: TaskInput,
        mode: Literal["train", "val", "test"] | None = None,
        resources_id: str | None = None,
        config: RolloutConfig | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> Rollout:
        """Persist a rollout in `queuing` state so runners can claim it later.

        !!! note
            Different from [`start_rollout()`][agentlightning.LightningStore.start_rollout],
            this method is called when the caller only wants to submit work for later scheduling.

        Implementations must generate a unique `rollout_id`, stamp `start_time` with
        the current time, default `config` to a fresh [`RolloutConfig`][agentlightning.RolloutConfig],
        and insert the rollout at the tail of the scheduling queue. No attempt is created yet.

        Args:
            input: Arbitrary task payload supplied by an algorithm.
            mode: Optional semantic mode indicator (`"train"`, `"val"`, `"test"`).
            resources_id: Resource snapshot used when a runner eventually executes the rollout.
            config: Fine-grained retry/timeout parameters to persist with the rollout.
            metadata: Free-form metadata stored verbatim with the rollout record.

        Returns:
            The stored [`Rollout`][agentlightning.Rollout] in `queuing` status.
        """
        raise NotImplementedError()

    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str | Literal["latest"],
        status: AttemptStatus | Unset = UNSET,
        worker_id: str | Unset = UNSET,
        last_heartbeat_time: float | Unset = UNSET,
        metadata: Optional[Dict[str, Any]] | Unset = UNSET,
    ) -> Attempt:
        """Update attempt bookkeeping such as status, worker ownership, and heartbeats.

        When `attempt_id` is `"latest"` the update must target the attempt with the highest
        `sequence_id`; otherwise it must target the specific attempt. Implementations should
        propagate status changes to the rollout (for example via [`propagate_status()`][agentlightning.store.utils.propagate_status])
        once the latest attempt transitions to a terminal state.

        Similar to [`update_rollout()`][agentlightning.LightningStore.update_rollout],
        parameters also default to the sentinel [`UNSET`][agentlightning.store.base.UNSET].

        Args:
            rollout_id: Identifier of the rollout whose attempt will be updated.
            attempt_id: Attempt identifier or `"latest"` as a convenience.
            status: Replacement attempt status. Terminal statuses must set `end_time`.
            worker_id: Identifier for the worker currently processing the attempt.
            last_heartbeat_time: Wall-clock timestamp (seconds) of the latest heartbeat/span.
            metadata: Replacement metadata dictionary.

        Returns:
            The updated attempt record.
        """
        raise NotImplementedError()