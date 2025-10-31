from typing import TYPE_CHECKING, Any, Generic, Iterator, Optional, Sequence, TypeVar

from .types import ParallelWorkerBase, AttemptedRollout
from .litagent import LitAgent
from .store import LightningStore

T_task = TypeVar("T_task")


# ===================================================================================================================
# agentlightning/runner/base.py
class Runner(ParallelWorkerBase, Generic[T_task]):
    """Abstract base class for long-running agent executors.

    Runner implementations coordinate [`LitAgent`][agentlightning.LitAgent]
    instances, acquire work from a [`LightningStore`][agentlightning.LightningStore],
    and emit [`Rollout`][agentlightning.Rollout] objects. Subclasses decide how
    to schedule work (polling, streaming, etc.) while this base class provides a
    minimal lifecycle contract.
    """

    def init(self, agent: LitAgent[T_task], **kwargs: Any) -> None:
        """Prepare the runner to execute tasks for `agent`.

        This method is called only once during the setup for all workers, not for each worker.

        Args:
            agent: Agent instance providing task-specific logic.
            **kwargs: Optional runner-specific configuration.
        """

# ===================================================================================================================
# agentlightning/runner/agent.py
""" 

核心逻辑在 _step_impl 中: 获取 store, agent
"""
class LitAgentRunner(Runner[T_task]):
    """Execute [`LitAgent`][agentlightning.LitAgent] tasks with tracing support.

    This runner manages the complete lifecycle of agent rollout execution,
    including task polling, resource management, tracing, and hooks. It supports
    both continuous iteration over tasks from the store and single-step execution.

    Attributes:
        worker_id: Identifier for the active worker process, if any.
    """
    def __init__(self, tracer: Tracer, max_rollouts: Optional[int] = None, poll_interval: float = 5.0) -> None:
        self._tracer = tracer
        self._max_rollouts = max_rollouts
        self._poll_interval = poll_interval

    def get_agent(self) -> LitAgent[T_task]:
        return self._agent
    def get_store(self) -> LightningStore:
        return self._store

    def init(self, agent: LitAgent[T_task], *, hooks: Optional[Sequence[Hook]] = None, **kwargs: Any) -> None:
        """Initialize the runner with the agent.

        This sets up the agent-runner relationship, registers hooks, and
        initializes the tracer."""
        self._agent = agent
        self._agent.set_runner(self)
        self._hooks = [*hooks] if hooks is not None else []
        self._tracer.init()

    def init_worker(self, worker_id: int, store: LightningStore, **kwargs: Any) -> None:
        """Initialize the runner for each worker with worker_id and store.

        This method is called once per worker in a distributed setup to provide
        the worker with its ID and store connection."""
        self._store = store
        self.worker_id = worker_id
        self._tracer.init_worker(worker_id)

    async def _step_impl(self, next_rollout: AttemptedRollout, raise_on_exception: bool = False) -> str:
        """Execute a single rollout implementation.

        This is the core method that handles the execution of a single rollout,
        including resource fetching, hook triggering, agent invocation, tracing,
        and result processing.

        Args:
            next_rollout: The rollout to execute, containing input data, mode,
                and resources information.
            raise_on_exception: If True, exceptions during rollout execution will
                be re-raised. If False, exceptions are logged but not propagated.
        """
        store = self.get_store()
        agent = self.get_agent()
        rollout_id = next_rollout.rollout_id
        try:
            async with self._tracer.trace_context(
                name=rollout_id, store=store, rollout_id=rollout_id, attempt_id=next_rollout.attempt.attempt_id
            ):
                # NOTE: This is the most costly step in the whole function
                # If the rollout method becomes unresponsive or timeouts, there is nothing we can do within the runner.
                # We might need some mechanisms in execution strategy to restart the runner. But that's a future work.
                if agent.is_async():
                    rollout_method = (
                        agent.training_rollout_async if next_rollout.mode == "train" else agent.validation_rollout_async
                    )
                    result = await rollout_method(
                        next_rollout.input, resources=resources_update.resources, rollout=next_rollout
                    )
            trace_spans = await self._post_process_rollout_result(next_rollout, result)
            last_reward = find_final_reward(trace_spans)
        finally:
            try:
                if has_exception:
                    # possibly timed out and cancelled?
                    await store.update_attempt(rollout_id, next_rollout.attempt.attempt_id, status="failed")
                else:
                    await store.update_attempt(rollout_id, next_rollout.attempt.attempt_id, status="succeeded")
        return rollout_id