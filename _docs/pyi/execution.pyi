import asyncio, multiprocessing
import logging
from typing import Protocol, Generic, Optional, Literal

from agentlightning.store.base import LightningStore

logger = logging.getLogger(__name__)

# ===================================================================================================================
# agentlightning/execution/events.py
class ExecutionEvent(Protocol):
    """Protocol capturing the cooperative stop contract shared by strategies.

    Implementations mirror the API of ``threading.Event`` and
    ``multiprocessing.Event`` so the rest of the execution layer can remain
    agnostic to the underlying concurrency primitive.

    Methods:

        set: Signal cancellation. The call must be idempotent.
        clear: Reset the event to the unsignaled state.
        is_set: Return ``True`` when cancellation has been requested.
        wait: Block until the event is signaled or an optional timeout elapses.
    """

    def set(self) -> None: ...
    def clear(self) -> None: ...
    def is_set(self) -> bool: ...
    def wait(self, timeout: Optional[float] = None) -> bool: ...

class MultiprocessingEvent:
    """Process-safe implementation of [`ExecutionEvent`][agentlightning.ExecutionEvent]."""

    __slots__ = ("_evt",)

    def __init__(self, *, ctx: Optional[BaseContext] = None) -> None:
        self._evt = (ctx or mp).Event()

    def set(self) -> None:
        self._evt.set()

    def clear(self) -> None:
        self._evt.clear()

    def is_set(self) -> bool:
        return self._evt.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._evt.wait(timeout)


# ===================================================================================================================
# agentlightning/execution/base.py
class AlgorithmBundle(Protocol):
    """Callable bundle produced by [`Trainer`][agentlightning.Trainer].

    Execution strategies treat the returned coroutine as opaque, only providing
    the shared store instance and cooperative stop event. Bundles typically
    encapsulate algorithm setup plus adapter and LLM proxy, etc.
    """

    async def __call__(self, store: LightningStore, event: ExecutionEvent) -> None:
        """Execute algorithm logic using ``store`` until completion or stop."""


class RunnerBundle(Protocol):
    """Callable bundle wrapping runner setup and the worker loop, as opposed to the
    [`AlgorithmBundle`][agentlightning.AlgorithmBundle]."""

    async def __call__(self, store: LightningStore, worker_id: int, event: ExecutionEvent) -> None:
        """Execute runner logic for ``worker_id`` using ``store`` and ``event``."""


class ExecutionStrategy:
    """Coordinate algorithm and runner bundles within a single process abstraction.

    Strategies decide how many worker bundles to launch, whether to communicate
    through shared memory or an HTTP boundary, and how to react to shutdown
    signals. They intentionally avoid inspecting the bundle internals; instead,
    each bundle remains responsible for its own scheduling semantics.

    !!! note
        Implementations must honor the [execute()][agentlightning.ExecutionStrategy.execute]
        contract by propagating `KeyboardInterrupt` and ensuring resources are
        released when an error occurs on either side of the algorithm/runner
        pair.
    """

    def execute(self, algorithm: AlgorithmBundle, runner: RunnerBundle, store: LightningStore) -> None:
        """Run the provided bundles using the configured orchestration model.

        Args:
            algorithm: Callable bundle responsible for algorithm execution.
            runner: Callable bundle for runner workers.
            store: Concrete [`LightningStore`][agentlightning.LightningStore]
                shared across bundles.

        Raises:
            NotImplementedError: Subclasses must provide the orchestration
                implementation.
        """

# ===================================================================================================================
# agentlightning/execution/client_server.py
""" ClientServerExecutionStrategy: 默认的执行策略
role: 指定执行角色，可以是 "algorithm"、"runner" 或 "both" (env.AGL_CURRENT_ROLE)

"""
class ClientServerExecutionStrategy(ExecutionStrategy):
    """Run algorithm and runner bundles as separate processes over HTTP.

    Execution Roles:

    - `"algorithm"`: Start [`LightningStoreServer`][agentlightning.LightningStoreServer]
      in-process and execute the algorithm bundle against it.
    - `"runner"`: Connect to an existing server with
      [`LightningStoreClient`][agentlightning.LightningStoreClient] and run the
      runner bundle locally (spawning multiple processes when requested).
    - `"both"`: Spawn runner processes first, then execute the algorithm and
      server on the same machine. This mode orchestrates the full loop locally.

    When `role == "both"` you may choose which side runs on the main process
    via `main_process`. The runner-on-main option is limited to
    `n_runners == 1` because each additional runner requires its own event
    loop and process.

    !!! warning
        When `main_process == "runner"` the algorithm and HTTP server execute
        in a child process. Store mutations remain isolated inside that process,
        so the original store instance passed to
        [execute()][agentlightning.ExecutionStrategy.execute] is not updated.

    Abort Model (four-step escalation):

    1. Cooperative stop. Every bundle receives a shared
       [`MultiprocessingEvent`][agentlightning.MultiprocessingEvent] (`stop_evt`).
       Any failure flips the event so peers can exit cleanly. Ctrl+C on the main
       process also sets the flag.
    2. KeyboardInterrupt synthesis. Remaining subprocesses receive ``SIGINT`` to
       trigger `KeyboardInterrupt` handlers.
    3. Termination. Stubborn processes are asked to ``terminate()``
       (`SIGTERM` on POSIX).
    4. Kill. As a last resort `kill()` is invoked (`SIGKILL` on POSIX).

    This mirrors the semantics implemented in
    [`SharedMemoryExecutionStrategy`][agentlightning.SharedMemoryExecutionStrategy]
    but adapts them to multiple processes and the HTTP client/server boundary.
    """
    def __init__(
        self,
        role: Literal["algorithm", "runner", "both"] | None = None,
        server_host: str | None = None,
        server_port: int | None = None,
        n_runners: int = 1,
        graceful_timeout: float = 5.0,
        terminate_timeout: float = 5.0,
        main_process: Literal["algorithm", "runner"] = "algorithm",
        managed_store: bool | None = None,
    ) -> None:
        self.role = role

    def execute(self, algorithm: AlgorithmBundle, runner: RunnerBundle, store: LightningStore) -> None:
        # Re-use the active multiprocessing context so the event and processes
        # agree on the start method (fork/spawn/forkserver).
        ctx = multiprocessing.get_context()
        stop_evt = MultiprocessingEvent(ctx=ctx)
        # Track spawned processes so we can enforce termination ordering and
        # surface non-zero exit codes back to the caller.
        processes: list[multiprocessing.Process] = []

        try:
            if self.role == "both":
                if self.main_process == "algorithm":
                    logger.info("Spawning runner processes...")
                    processes = self._spawn_runners(runner, store, stop_evt, ctx=ctx)
                    try:
                        logger.info("Running algorithm...")
                        asyncio.run(self._execute_algorithm(algorithm, store, stop_evt))
                    finally:
                        # Always request the runner side to unwind once the
                        # algorithm/server portion finishes (successfully or not).
                        stop_evt.set()
                else:  # main_process == "runner"
                    if self.n_runners > 1:
                        raise ValueError("main_process='runner' requires n_runners to be 1")

                    logger.info("Spawning algorithm process...")
                    algorithm_process = self._spawn_algorithm_process(algorithm, store, stop_evt, ctx=ctx)
                    processes = [algorithm_process]

                    # Run the lone runner cooperatively in-process so users can
                    # attach a debugger. The algorithm + HTTP server live in
                    # the background process spawned above (the provided
                    # store must therefore be picklable when using spawn).
                    logger.info("Running runner...")
                    asyncio.run(self._execute_runner(runner, 0, store, stop_evt))

                    # Wait for the algorithm process to finish.
                    algorithm_process.join()
            else:
                pass
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received; initiating shutdown")
            stop_evt.set()
            keyboard_interrupt = True
        finally:
            logger.info("Shutting down subprocesses")
            self._shutdown_processes(processes, stop_evt)

    async def _execute_algorithm(
        self, algorithm: AlgorithmBundle, store: LightningStore, stop_evt: ExecutionEvent
    ) -> None:
        try:
            await algorithm(wrapper_store, stop_evt)
        except KeyboardInterrupt:
            stop_evt.set()
            raise

    async def _execute_runner(
        self,
        runner: RunnerBundle,
        worker_id: int,
        store: LightningStore,
        stop_evt: ExecutionEvent,
    ) -> None:
        try:
            await runner(client_store, worker_id, stop_evt)
