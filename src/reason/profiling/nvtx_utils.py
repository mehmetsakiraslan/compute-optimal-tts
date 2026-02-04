"""NVTX (NVIDIA Tools Extension) utilities for profiling test-time compute.

Provides context managers and decorators for instrumenting code with NVTX markers
that are visible in NVIDIA Nsight Systems profiler.

Color scheme:
- LM (Language Model): Green
- RM (Reward Model): Yellow
- Beam Selection: Red
- Tree Expansion: Blue
- HTTP Requests: Orange
"""

import functools
from contextlib import contextmanager
from typing import Optional

# Try to import nvtx, provide no-op fallbacks if unavailable
try:
    import nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False


class NVTXColors:
    """NVTX color codes for different profiling categories.

    Colors are ARGB format (32-bit integer).
    """
    # Primary colors for different operations
    LM_GREEN = 0xFF00FF00        # Language model operations
    RM_YELLOW = 0xFFFFFF00       # Reward model operations
    BEAM_RED = 0xFFFF0000        # Beam selection/search
    TREE_BLUE = 0xFF0000FF       # Tree expansion
    HTTP_ORANGE = 0xFFFF8000     # HTTP/network requests

    # Additional colors for finer granularity
    DEPTH_CYAN = 0xFF00FFFF      # Per-depth iteration
    BATCH_MAGENTA = 0xFFFF00FF   # Batch processing
    ENV_PURPLE = 0xFF8000FF      # Environment operations


@contextmanager
def nvtx_range(name: str, color: Optional[int] = None, domain: str = "tts"):
    """Context manager for NVTX range annotation.

    Usage:
        with nvtx_range("beam_search_total", NVTXColors.BEAM_RED):
            # code to profile

    Args:
        name: Name of the range that appears in Nsight Systems
        color: ARGB color code (use NVTXColors constants)
        domain: NVTX domain for grouping related markers
    """
    if not NVTX_AVAILABLE:
        yield
        return

    # Create range with optional color
    if color is not None:
        rng = nvtx.start_range(message=name, color=color, domain=domain)
    else:
        rng = nvtx.start_range(message=name, domain=domain)

    try:
        yield
    finally:
        nvtx.end_range(rng)


def nvtx_annotate(name: Optional[str] = None, color: Optional[int] = None, domain: str = "tts"):
    """Decorator for NVTX annotation of functions.

    Usage:
        @nvtx_annotate("expand_leaf_node", NVTXColors.TREE_BLUE)
        def _expand_leaf_node(self, node, env, rm_call):
            # function body

    Args:
        name: Name of the range (defaults to function name)
        color: ARGB color code (use NVTXColors constants)
        domain: NVTX domain for grouping related markers
    """
    def decorator(func):
        range_name = name if name is not None else func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with nvtx_range(range_name, color, domain):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def nvtx_mark(message: str, color: Optional[int] = None, domain: str = "tts"):
    """Place an instantaneous NVTX marker (useful for events).

    Args:
        message: Marker message
        color: ARGB color code
        domain: NVTX domain
    """
    if not NVTX_AVAILABLE:
        return

    if color is not None:
        nvtx.mark(message=message, color=color, domain=domain)
    else:
        nvtx.mark(message=message, domain=domain)
