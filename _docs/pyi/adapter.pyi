from typing import Generic, List, TypeVar

from .types import Span

T_from = TypeVar("T_from")
T_to = TypeVar("T_to")

# ===================================================================================================================
# agentlightning/adapter/base.py
class Adapter(Generic[T_from, T_to]):
    """Base class for synchronous adapters that convert data from one format to another.

    The class defines a minimal protocol so that adapters can be treated like callables while
    still allowing subclasses to supply the concrete transformation logic.

    !!! note
        Subclasses must override [`adapt()`][agentlightning.Adapter.adapt] to provide
        the actual conversion.

    Type Variables:

        T_from: Source data type supplied to the adapter.

        T_to: Target data type produced by the adapter.

    Examples:
        >>> class IntToStrAdapter(Adapter[int, str]):
        ...     def adapt(self, source: int) -> str:
        ...         return str(source)
        ...
        >>> adapter = IntToStrAdapter()
        >>> adapter(42)
        '42'
    """

class TraceAdapter(Adapter[List[Span], T_to], Generic[T_to]):
    """Base class for adapters that convert trace spans into other formats.

    This class specializes [`Adapter`][agentlightning.Adapter] for working with
    [`Span`][agentlightning.Span] instances emitted by Agent Lightning instrumentation.
    Subclasses receive entire trace slices and return a format suited for the downstream consumer,
    for example reinforcement learning training data or observability metrics.
    """
