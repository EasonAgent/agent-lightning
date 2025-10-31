from typing import Optional, Any, TypeVar
import functools

from agentlightning.adapter import TraceAdapter, TracerTraceToTriplet
from agentlightning.algorithm import Algorithm, Baseline, FastAlgorithm
from agentlightning.client import AgentLightningClient
from agentlightning.execution.base import ExecutionStrategy
from agentlightning.execution.client_server import ClientServerExecutionStrategy
from agentlightning.execution.events import ExecutionEvent
from agentlightning.litagent import LitAgent
from agentlightning.llm_proxy import LLMProxy
from agentlightning.runner import LitAgentRunner, Runner
from agentlightning.store.base import LightningStore
from agentlightning.store.memory import InMemoryLightningStore
from agentlightning.tracer.agentops import AgentOpsTracer
from agentlightning.tracer.base import Tracer
from agentlightning.types import Dataset, Hook, NamedResources

from agentlightning.trainer.legacy import TrainerLegacy

T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")

# agentlightning/trainer/trainer.py
class Trainer(TrainerLegacy):
    """High-level orchestration layer that wires Algorithm <-> Runner <-> Store.

    A [`Trainer`][agentlightning.Trainer] packages the moving parts of Agent-Lightning's
    training loop into a single entry point:

    * **Algorithm lifecycle:** Instantiates or accepts an [`Algorithm`][agentlightning.Algorithm],
      attaches the current [`LightningStore`][agentlightning.LightningStore], adapter, and
      initial resources, then executes the algorithm role inside the configured execution strategy.
    * **Runner fleet:** Spawns one or more [`Runner`][agentlightning.Runner] instances (defaulting
      to [`LitAgentRunner`][agentlightning.LitAgentRunner]) that hydrate a [`LitAgent`][agentlightning.LitAgent],
      claim rollouts, stream spans, and respect graceful termination signals from the execution strategy.
    * **Execution strategy:** Delegates process management to an
      [`ExecutionStrategy`][agentlightning.ExecutionStrategy] (shared memory, client/server, etc.),
      so advanced users can swap orchestration backends without changing trainer code.
    * **Telemetry plumbing:** Ensures tracers, adapters, and optional [`LLMProxy`][agentlightning.LLMProxy]
      are wired into both algorithm and runners so telemetry flows back into the store.

    The trainer exposes two convenience entry points:
    [`fit()`][agentlightning.Trainer.fit] for full training and
    [`dev()`][agentlightning.Trainer.dev] for fast, reproducible dry-runs. See the
    [Train the First Agent](../how-to/train-first-agent.md) and
    [Write the First Algorithm](../how-to/write-first-algorithm.md) tutorials for the broader context.
    """

    algorithm: Optional[Algorithm]
    """An instance of [`Algorithm`][agentlightning.Algorithm] to use for training."""
    store: LightningStore
    """An instance of [`LightningStore`][agentlightning.LightningStore] to use for storing tasks and traces."""
    runner: Runner[Any]
    """An instance of [`Runner`][agentlightning.Runner] to use for running the agent."""
    strategy: ExecutionStrategy
    """An instance of [`ExecutionStrategy`][agentlightning.ExecutionStrategy] to use for spawning the algorithm and runners."""

    def fit(
        self,
        agent: LitAgent[T_co],
        train_dataset: Optional[Dataset[T_co]] = None,
        *,
        val_dataset: Optional[Dataset[T_co]] = None,
    ) -> None:
        """Execute the full algorithm/runner training loop.

        [`Trainer.fit`][agentlightning.Trainer.fit] packages the algorithm and runner bundles,
        then hands them to the active [`ExecutionStrategy`][agentlightning.ExecutionStrategy].
        The strategy rarely returns until:

        * The algorithm exhausts the dataset(s) and stops enqueuing rollouts.
        * `max_rollouts` causes individual runners to exit.
        * An exception or interrupt cancels the shared [`ExecutionEvent`][agentlightning.ExecutionEvent].

        Args:
            agent: [`LitAgent`][agentlightning.LitAgent] implementation executed by runners.
            train_dataset: Optional iterable of rollout inputs consumed by the algorithm.
            val_dataset: Optional iterable consumed by validation passes.
        """
        agent.set_trainer(self)
        algorithm_bundle = functools.partial(
            self._algorithm_bundle,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            algorithm=self.algorithm,
        )
        runner_bundle = functools.partial(self._runner_bundle, agent=agent)
        self.strategy.execute(algorithm_bundle, runner_bundle, self.store)

    def dev(
        self,
        agent: LitAgent[T_co],
        train_dataset: Optional[Dataset[T_co]] = None,
        *,
        val_dataset: Optional[Dataset[T_co]] = None,
    ) -> None:
        """Exercise the infrastructure using a fast, synchronous algorithm.

        [`Trainer.dev`][agentlightning.Trainer.dev] mirrors [`fit()`][agentlightning.Trainer.fit] but
        insists on an [`Algorithm`][agentlightning.Algorithm] subtype that also derives from
        [`FastAlgorithm`][agentlightning.FastAlgorithm]. This keeps the loop responsive for
        debugging while still touching the same store, runners, hooks, and tracer plumbing.

        If no algorithm is provided, a default [`Baseline`][agentlightning.Baseline] algorithm will be used.

        Args:
            agent: [`LitAgent`][agentlightning.LitAgent] implementation to execute.
            train_dataset: Optional iterable passed to the algorithm.
            val_dataset: Optional iterable passed to the algorithm.

        Raises:
            TypeError: If the configured algorithm does not inherit from `FastAlgorithm`.
        """