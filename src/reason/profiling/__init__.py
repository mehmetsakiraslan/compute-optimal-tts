"""Profiling utilities for test-time scaling analysis.

This package provides:
- NVTX utilities for GPU profiling with Nsight Systems
- Execution graph tracing for CPU-side timing analysis
- Tree structure tracing for beam search visualization
- Chrome trace export for visualization

Note: traced_callers, trace_export, and tree-related modules are imported lazily
to avoid circular imports. Import them directly when needed:
    from reason.profiling.traced_callers import TracedVLLMRemoteCaller
    from reason.profiling.trace_export import export_chrome_trace
    from reason.profiling.tree_tracer import TreeTracer
    from reason.profiling.instrumented_tree import InstrumentedSearchTree
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
    """Lazy import for modules that have dependencies on lm_call/rm_call or guided_search."""
    if name in ("TracedVLLMRemoteCaller", "TracedRMRemoteCaller",
                "wrap_existing_rm_call", "wrap_existing_lm_call"):
        from reason.profiling import traced_callers
        return getattr(traced_callers, name)
    elif name in ("export_chrome_trace", "export_structured_json",
                  "export_timeline_csv", "print_timeline_ascii"):
        from reason.profiling import trace_export
        return getattr(trace_export, name)
    elif name in ("TreeTracer", "TreeNodeEvent", "DepthTimingEvent", "NodeStatus"):
        from reason.profiling import tree_tracer
        return getattr(tree_tracer, name)
    elif name == "InstrumentedSearchTree":
        from reason.profiling import instrumented_tree
        return getattr(instrumented_tree, name)
    elif name in ("build_tree_json", "render_ascii_tree", "render_html_tree",
                  "export_tree_json", "export_ascii_tree", "export_html_tree"):
        from reason.profiling import tree_visualizer
        return getattr(tree_visualizer, name)
    elif name == "build_tree_from_output":
        from reason.profiling import offline_tree_builder
        return getattr(offline_tree_builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
