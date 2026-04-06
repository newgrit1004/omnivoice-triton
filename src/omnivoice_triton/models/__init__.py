"""Model runners and patching for OmniVoice."""

from omnivoice_triton.models.base_runner import BaseRunner
from omnivoice_triton.models.faster_runner import FasterRunner
from omnivoice_triton.models.patching import apply_sage_attention, apply_triton_kernels
from omnivoice_triton.models.triton_faster_runner import TritonFasterRunner
from omnivoice_triton.models.triton_runner import TritonRunner

__all__ = [
    "BaseRunner",
    "FasterRunner",
    "TritonRunner",
    "TritonFasterRunner",
    "apply_triton_kernels",
    "apply_sage_attention",
    "get_runner_class",
    "create_runner",
    "ALL_RUNNER_NAMES",
]

_RUNNER_MAP: dict[str, type] = {
    "base": BaseRunner,
    "triton": TritonRunner,
    "faster": FasterRunner,
    "hybrid": TritonFasterRunner,
}

ALL_RUNNER_NAMES: list[str] = [
    "base",
    "triton",
    "faster",
    "hybrid",
]


def get_runner_class(name: str) -> type:
    """Look up a runner class by short name."""
    key = name.lower()
    if key not in _RUNNER_MAP:
        available = ", ".join(sorted(_RUNNER_MAP))
        msg = f"Unknown runner '{name}'. Available: {available}"
        raise KeyError(msg)
    return _RUNNER_MAP[key]


def create_runner(name: str, **kwargs) -> BaseRunner:  # type: ignore[return-value]
    """Create a runner instance by name."""
    cls = get_runner_class(name)
    return cls(**kwargs)  # type: ignore[call-arg]
