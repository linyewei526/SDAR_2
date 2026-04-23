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


class _TimedNvtxRange:
    def __init__(self, message: str, stream=None):
        self.message = message
        self.stream = stream
        self._start_event = None
        self._end_event = None
        self._nvtx_pushed = False

    def __enter__(self):
        if _NVTX_ENABLED:
            import torch.cuda.nvtx as torch_nvtx

            torch_nvtx.range_push(self.message)
            self._nvtx_pushed = True

        if self._should_record_timing():
            import torch

            self._stream = self.stream or torch.cuda.current_stream()
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record(self._stream)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._start_event is not None:
            self._end_event.record(self._stream)
            try:
                from baseline.sdar_runtime_trace import record_timing_range

                record_timing_range(self.message, self._start_event, self._end_event)
            except ImportError:
                pass

        if self._nvtx_pushed:
            import torch.cuda.nvtx as torch_nvtx

            torch_nvtx.range_pop()

        return False

    def _should_record_timing(self) -> bool:
        try:
            import torch
            from baseline.sdar_runtime_trace import is_latency_recording_enabled
        except ImportError:
            return False

        return torch.cuda.is_available() and is_latency_recording_enabled()


def nvtx_range(message: str, stream=None):
    """
    Return a real NVTX range context when enabled, otherwise a no-op context.

    When SDAR latency summary recording is enabled, the same context also
    records CUDA event timestamps aligned with the NVTX range label.
    """
    if not _NVTX_ENABLED:
        try:
            from baseline.sdar_runtime_trace import is_latency_recording_enabled
            import torch
        except ImportError:
            return nullcontext()

        if not (torch.cuda.is_available() and is_latency_recording_enabled()):
            return nullcontext()

    return _TimedNvtxRange(message, stream=stream)
