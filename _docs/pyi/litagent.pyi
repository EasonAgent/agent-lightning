from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, TypeVar
import weakref

from .trainer import Trainer
from .runner import Runner

T = TypeVar("T")

# ===================================================================================================================
# agentlightning/litagent/litagent.py
class LitAgent(Generic[T]):
    """Base class for implementing agent rollouts.

    Subclasses override the rollout methods to process tasks while the trainer and
    runner infrastructure manages orchestration, tracing, and persistence.
    """

    def set_trainer(self, trainer: Trainer) -> None:
        """Attach the trainer responsible for orchestration.

        Args:
            trainer: [`Trainer`][agentlightning.Trainer] that manages the agent.
        """
        self._trainer_ref = weakref.ref(trainer)
    def set_runner(self, runner: Runner[T]) -> None:
        """Attach the runner responsible for executing rollouts.

        Args:
            runner: [`Runner`][agentlightning.Runner] coordinating execution.
        """
        self._runner_ref = weakref.ref(runner)

    async def training_rollout_async(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutRawResult:
        """Process a single training task asynchronously.

        By default, this method delegates to
        [`rollout_async`][agentlightning.LitAgent.rollout_async].
        """
        return await self.rollout_async(task, resources, rollout)

    async def validation_rollout_async(self, task: T, resources: NamedResources, rollout: Rollout) -> RolloutRawResult:
        """Process a single validation task asynchronously.

        Override this method when validation should differ from training. The default
        implementation delegates to
        [`training_rollout_async`][agentlightning.LitAgent.training_rollout_async].
        """
        return await self.rollout_async(task, resources, rollout)
