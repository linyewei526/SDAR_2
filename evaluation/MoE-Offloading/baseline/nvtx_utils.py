"""
Utilities for globally gating NVTX instrumentation.

Default behavior keeps NVTX disabled so bare baseline runs avoid profiler
annotation overhead unless explicitly requested.
"""

from contextlib import nullcontext


_NVTX_ENABLED = False


def set_nvtx_enabled(enabled: bool) -> None:
    """Enable or disable NVTX range emission process-wide."""
    global _NVTX_ENABLED
    _NVTX_ENABLED = bool(enabled)


def is_nvtx_enabled() -> bool:
    return _NVTX_ENABLED


def nvtx_range(message: str):
    """
    Return a real NVTX range context when enabled, otherwise a no-op context.
    """
    if not _NVTX_ENABLED:
        return nullcontext()

    import torch.cuda.nvtx as torch_nvtx

    return torch_nvtx.range(message)
