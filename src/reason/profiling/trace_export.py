"""Export functions for execution trace data.

This module provides functions to export execution traces to:
1. Chrome Trace Format (for chrome://tracing or Perfetto UI)
2. Structured JSON (for programmatic analysis)
"""

import json
from typing import Dict, List, Optional

from reason.profiling.execution_tracer import (
    ExecutionTracer,
    ExecutionEvent,
    EventType,
)


# Color mapping for Chrome trace visualization (in AARRGGBB format as integer)
CHROME_TRACE_COLORS = {
    EventType.LM_CALL: "good",  # Green
    EventType.RM_CALL: "bad",  # Yellow/Orange
    EventType.TREE_SELECT: "terrible",  # Red
    EventType.TREE_EXPAND: "generic_work",  # Blue
    EventType.DEPTH_ITERATION: "thread_state_running",  # Cyan
}

# Model to thread ID mapping for Chrome trace
MODEL_TO_TID = {
    "Policy": 1,
    "Value": 2,
    "Tree": 3,
}


def export_chrome_trace(
    tracer: Optional[ExecutionTracer] = None,
    output_path: Optional[str] = None,
) -> Dict:
    """Export execution trace to Chrome Trace Format.

    The output can be loaded in chrome://tracing or https://ui.perfetto.dev/
    for visualization.

    Args:
        tracer: ExecutionTracer instance (uses singleton if not provided)
        output_path: Path to write JSON file (returns dict if not provided)

    Returns:
        Chrome trace format dictionary
    """
    if tracer is None:
        tracer = ExecutionTracer.get_instance()

    events = tracer.get_events()

    chrome_events = []
    flow_id_counter = 0

    # Create process and thread metadata events
    chrome_events.append({
        "name": "process_name",
        "ph": "M",
        "pid": 1,
        "args": {"name": "Test-Time Scaling"}
    })

    for model_name, tid in MODEL_TO_TID.items():
        chrome_events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": 1,
            "tid": tid,
            "args": {"name": f"{model_name} Model"}
        })

    # Convert events to Chrome trace format
    for event in events:
        tid = MODEL_TO_TID.get(event.model, 0)

        # Duration event (begin + end)
        chrome_events.append({
            "name": f"{event.event_type.value}",
            "cat": event.model,
            "ph": "B",  # Begin
            "ts": event.start_time,
            "pid": 1,
            "tid": tid,
            "args": {
                "event_id": event.event_id,
                "depth": event.depth,
                **event.metadata,
            },
            "cname": CHROME_TRACE_COLORS.get(event.event_type, "generic_work"),
        })

        chrome_events.append({
            "name": f"{event.event_type.value}",
            "cat": event.model,
            "ph": "E",  # End
            "ts": event.end_time,
            "pid": 1,
            "tid": tid,
        })

        # Add flow arrows for dependencies
        if event.triggered_by is not None:
            # Find parent event
            parent_event = None
            for e in events:
                if e.event_id == event.triggered_by:
                    parent_event = e
                    break

            if parent_event:
                parent_tid = MODEL_TO_TID.get(parent_event.model, 0)

                # Flow start (at end of parent)
                chrome_events.append({
                    "name": "dependency",
                    "cat": "flow",
                    "ph": "s",  # Flow start
                    "ts": parent_event.end_time,
                    "pid": 1,
                    "tid": parent_tid,
                    "id": flow_id_counter,
                    "bp": "e",  # Bind to enclosing slice end
                })

                # Flow end (at start of child)
                chrome_events.append({
                    "name": "dependency",
                    "cat": "flow",
                    "ph": "f",  # Flow finish
                    "ts": event.start_time,
                    "pid": 1,
                    "tid": tid,
                    "id": flow_id_counter,
                    "bp": "e",
                })

                flow_id_counter += 1

    # Add stall time indicators as separate events
    for event in events:
        if event.stall_time > 0:
            tid = MODEL_TO_TID.get(event.model, 0)
            stall_start = event.start_time - event.stall_time

            chrome_events.append({
                "name": "stall",
                "cat": "stall",
                "ph": "B",
                "ts": stall_start,
                "pid": 1,
                "tid": tid,
                "args": {"stall_time_us": event.stall_time},
                "cname": "grey",
            })

            chrome_events.append({
                "name": "stall",
                "cat": "stall",
                "ph": "E",
                "ts": event.start_time,
                "pid": 1,
                "tid": tid,
            })

    # Add depth markers
    depths = sorted(set(e.depth for e in events))
    for depth in depths:
        depth_events = [e for e in events if e.depth == depth]
        if depth_events:
            depth_start = min(e.start_time for e in depth_events)
            depth_end = max(e.end_time for e in depth_events)

            chrome_events.append({
                "name": f"Depth {depth}",
                "cat": "depth",
                "ph": "B",
                "ts": depth_start,
                "pid": 1,
                "tid": 0,  # Separate thread for depth markers
                "cname": "cq_build_running",
            })

            chrome_events.append({
                "name": f"Depth {depth}",
                "cat": "depth",
                "ph": "E",
                "ts": depth_end,
                "pid": 1,
                "tid": 0,
            })

    # Add thread name for depth markers
    chrome_events.append({
        "name": "thread_name",
        "ph": "M",
        "pid": 1,
        "tid": 0,
        "args": {"name": "Depth Progress"}
    })

    trace_data = {"traceEvents": chrome_events}

    if output_path:
        with open(output_path, "w") as f:
            json.dump(trace_data, f, indent=2)

    return trace_data


def export_structured_json(
    tracer: Optional[ExecutionTracer] = None,
    output_path: Optional[str] = None,
) -> Dict:
    """Export execution trace to structured JSON for programmatic analysis.

    Args:
        tracer: ExecutionTracer instance (uses singleton if not provided)
        output_path: Path to write JSON file (returns dict if not provided)

    Returns:
        Structured JSON dictionary with events, summary, and dependency graph
    """
    if tracer is None:
        tracer = ExecutionTracer.get_instance()

    events = tracer.get_events()
    summary = tracer.get_summary()

    # Build dependency adjacency list
    dependency_graph = {}
    for event in events:
        event_id = str(event.event_id)
        dependency_graph[event_id] = {
            "triggered_by": event.triggered_by,
            "triggers": event.triggers,
        }

    # Group events by type
    events_by_type = {}
    for event in events:
        event_type = event.event_type.value
        if event_type not in events_by_type:
            events_by_type[event_type] = []
        events_by_type[event_type].append(event.to_dict())

    # Group events by depth
    events_by_depth = {}
    for event in events:
        depth = str(event.depth)
        if depth not in events_by_depth:
            events_by_depth[depth] = []
        events_by_depth[depth].append(event.event_id)

    # Compute per-depth statistics
    depth_stats = {}
    for depth in sorted(set(e.depth for e in events)):
        depth_events = [e for e in events if e.depth == depth]
        lm_events = [e for e in depth_events if e.event_type == EventType.LM_CALL]
        rm_events = [e for e in depth_events if e.event_type == EventType.RM_CALL]

        depth_stats[str(depth)] = {
            "total_events": len(depth_events),
            "lm_calls": len(lm_events),
            "rm_calls": len(rm_events),
            "lm_time_ms": sum(e.duration for e in lm_events) / 1000,
            "rm_time_ms": sum(e.duration for e in rm_events) / 1000,
            "total_stall_ms": sum(e.stall_time for e in depth_events) / 1000,
        }

    output = {
        "version": "1.0",
        "summary": summary,
        "depth_statistics": depth_stats,
        "events": [e.to_dict() for e in events],
        "events_by_type": events_by_type,
        "events_by_depth": events_by_depth,
        "dependency_graph": dependency_graph,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

    return output


def export_timeline_csv(
    tracer: Optional[ExecutionTracer] = None,
    output_path: Optional[str] = None,
) -> str:
    """Export execution trace to CSV format for spreadsheet analysis.

    Args:
        tracer: ExecutionTracer instance (uses singleton if not provided)
        output_path: Path to write CSV file (returns string if not provided)

    Returns:
        CSV string with event data
    """
    if tracer is None:
        tracer = ExecutionTracer.get_instance()

    events = tracer.get_events()

    lines = ["event_id,event_type,model,depth,start_time_ms,end_time_ms,duration_ms,stall_time_ms,triggered_by"]

    for event in events:
        line = ",".join([
            str(event.event_id),
            event.event_type.value,
            event.model,
            str(event.depth),
            f"{event.start_time / 1000:.3f}",
            f"{event.end_time / 1000:.3f}",
            f"{event.duration / 1000:.3f}",
            f"{event.stall_time / 1000:.3f}",
            str(event.triggered_by) if event.triggered_by is not None else "",
        ])
        lines.append(line)

    csv_content = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(csv_content)

    return csv_content


def print_timeline_ascii(
    tracer: Optional[ExecutionTracer] = None,
    width: int = 80,
) -> None:
    """Print an ASCII art timeline of the execution.

    Args:
        tracer: ExecutionTracer instance (uses singleton if not provided)
        width: Width of the timeline in characters
    """
    if tracer is None:
        tracer = ExecutionTracer.get_instance()

    events = tracer.get_events()
    if not events:
        print("No events to display.")
        return

    # Get time range
    min_time = min(e.start_time for e in events)
    max_time = max(e.end_time for e in events)
    time_range = max_time - min_time

    if time_range == 0:
        time_range = 1

    def time_to_col(t):
        return int((t - min_time) / time_range * (width - 20))

    print("\n" + "=" * width)
    print("EXECUTION TIMELINE")
    print("=" * width)

    # Print header
    print(f"{'Model':<12} | {'Timeline':<{width-15}}")
    print("-" * width)

    # Group by model
    for model in ["Policy", "Value", "Tree"]:
        model_events = [e for e in events if e.model == model]
        if not model_events:
            continue

        # Create timeline string
        timeline = [" "] * (width - 14)

        for event in model_events:
            start_col = time_to_col(event.start_time)
            end_col = time_to_col(event.end_time)
            end_col = max(start_col + 1, end_col)

            char = "█" if event.event_type in (EventType.LM_CALL, EventType.RM_CALL) else "▒"

            for i in range(start_col, min(end_col, len(timeline))):
                timeline[i] = char

        print(f"{model:<12} |{''.join(timeline)}")

    # Print time scale
    print("-" * width)
    print(f"{'Time (ms)':<12} | 0{' ' * (width - 30)}{time_range/1000:.1f}")
    print("=" * width)
