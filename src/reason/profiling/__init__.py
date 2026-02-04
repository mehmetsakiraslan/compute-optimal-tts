"""Profiling utilities for test-time scaling analysis.

This package provides:
- NVTX utilities for GPU profiling with Nsight Systems
- Execution graph tracing for CPU-side timing analysis
- Chrome trace export for visualization

Note: traced_callers and trace_export are imported lazily to avoid circular imports.
Import them directly when needed:
    from reason.profiling.traced_callers import TracedVLLMRemoteCaller
    from reason.profiling.trace_export import export_chrome_trace
"""

# These are safe to import at module level (no dependencies on lm_call/rm_call)
from reason.profiling.nvtx_utils import nvtx_range, nvtx_annotate, NVTXColors
from reason.profiling.execution_tracer import (
    ExecutionTracer,
    ExecutionEvent,
    EventType,
)

__all__ = [
    # NVTX utilities
    "nvtx_range",
    "nvtx_annotate",
    "NVTXColors",
    # Execution tracer
    "ExecutionTracer",
    "ExecutionEvent",
    "EventType",
]


def __getattr__(name):
    """Lazy import for modules that have dependencies on lm_call/rm_call."""
    if name in ("TracedVLLMRemoteCaller", "TracedRMRemoteCaller",
                "wrap_existing_rm_call", "wrap_existing_lm_call"):
        from reason.profiling import traced_callers
        return getattr(traced_callers, name)
    elif name in ("export_chrome_trace", "export_structured_json",
                  "export_timeline_csv", "print_timeline_ascii"):
        from reason.profiling import trace_export
        return getattr(trace_export, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
