"""Execution graph tracer for profiling test-time compute.

This module provides a thread-safe singleton tracer that collects execution events
from LM/RM calls and tree operations, computing dependencies and stall times.
"""

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class EventType(Enum):
    """Types of events in the execution trace."""
    LM_CALL = "lm_call"
    RM_CALL = "rm_call"
    TREE_SELECT = "tree_select"
    TREE_EXPAND = "tree_expand"
    DEPTH_ITERATION = "depth_iteration"


@dataclass
class ExecutionEvent:
    """A single execution event in the trace.

    Attributes:
        event_id: Unique identifier for this event
        event_type: Type of event (LM_CALL, RM_CALL, etc.)
        model: Which model generated this event ("Policy"/"Value"/"Tree")
        start_time: Start time in microseconds relative to trace start
        end_time: End time in microseconds relative to trace start
        duration: Duration of the event in microseconds
        triggered_by: Parent event ID that triggered this event
        triggers: List of child event IDs triggered by this event
        stall_time: Time waiting for dependencies in microseconds
        depth: Beam search depth at which this event occurred
        metadata: Additional information (tokens, batch_size, etc.)
    """
    event_id: int
    event_type: EventType
    model: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    triggered_by: Optional[int] = None
    triggers: List[int] = field(default_factory=list)
    stall_time: float = 0.0
    depth: int = 0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "model": self.model,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "triggered_by": self.triggered_by,
            "triggers": self.triggers,
            "stall_time": self.stall_time,
            "depth": self.depth,
            "metadata": self.metadata,
        }


class ExecutionTracer:
    """Thread-safe singleton for collecting execution events.

    Usage:
        tracer = ExecutionTracer.get_instance()
        tracer.start_trace()

        event = tracer.start_event(EventType.LM_CALL, "Policy", depth=0)
        # ... do work ...
        tracer.end_event(event)

        tracer.end_trace()
        tracer.compute_stall_times()
    """

    _instance: Optional["ExecutionTracer"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ExecutionTracer":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._events: List[ExecutionEvent] = []
        self._event_counter: int = 0
        self._trace_start_time: float = 0.0
        self._trace_end_time: float = 0.0
        self._is_tracing: bool = False
        self._current_depth: int = 0
        self._events_lock: threading.Lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ExecutionTracer":
        """Get the singleton tracer instance."""
        return cls()

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None

    def start_trace(self) -> None:
        """Begin a new execution trace."""
        with self._events_lock:
            self._events = []
            self._event_counter = 0
            self._trace_start_time = time.perf_counter() * 1_000_000  # microseconds
            self._is_tracing = True
            self._current_depth = 0

    def end_trace(self) -> None:
        """End the current execution trace."""
        with self._events_lock:
            self._trace_end_time = time.perf_counter() * 1_000_000
            self._is_tracing = False

    def set_current_depth(self, depth: int) -> None:
        """Set the current beam search depth."""
        self._current_depth = depth

    def start_event(
        self,
        event_type: EventType,
        model: str,
        depth: Optional[int] = None,
        triggered_by: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> ExecutionEvent:
        """Start a new execution event.

        Args:
            event_type: Type of the event
            model: Model name ("Policy", "Value", or "Tree")
            depth: Beam search depth (uses current depth if not provided)
            triggered_by: Parent event ID
            metadata: Additional event metadata

        Returns:
            The newly created ExecutionEvent
        """
        if not self._is_tracing:
            return ExecutionEvent(
                event_id=-1,
                event_type=event_type,
                model=model,
            )

        with self._events_lock:
            event_id = self._event_counter
            self._event_counter += 1

            start_time = time.perf_counter() * 1_000_000 - self._trace_start_time

            event = ExecutionEvent(
                event_id=event_id,
                event_type=event_type,
                model=model,
                start_time=start_time,
                depth=depth if depth is not None else self._current_depth,
                triggered_by=triggered_by,
                metadata=metadata or {},
            )

            self._events.append(event)

            # Update parent's triggers list
            if triggered_by is not None:
                for e in self._events:
                    if e.event_id == triggered_by:
                        e.triggers.append(event_id)
                        break

            return event

    def end_event(self, event: ExecutionEvent, metadata_update: Optional[Dict] = None) -> None:
        """End an execution event.

        Args:
            event: The event to end
            metadata_update: Additional metadata to add
        """
        if not self._is_tracing or event.event_id == -1:
            return

        with self._events_lock:
            end_time = time.perf_counter() * 1_000_000 - self._trace_start_time
            event.end_time = end_time
            event.duration = end_time - event.start_time

            if metadata_update:
                event.metadata.update(metadata_update)

    def compute_stall_times(self) -> None:
        """Compute stall times for all events based on dependencies.

        Stall time is the gap between when an event could have started
        (when its dependencies completed) and when it actually started.

        For RM events: stall = gap between LM completion and RM start (same depth)
        For LM events: stall = gap between previous RM completion and LM start
        """
        with self._events_lock:
            # Group events by depth and type
            lm_events_by_depth: Dict[int, List[ExecutionEvent]] = {}
            rm_events_by_depth: Dict[int, List[ExecutionEvent]] = {}

            for event in self._events:
                if event.event_type == EventType.LM_CALL:
                    if event.depth not in lm_events_by_depth:
                        lm_events_by_depth[event.depth] = []
                    lm_events_by_depth[event.depth].append(event)
                elif event.event_type == EventType.RM_CALL:
                    if event.depth not in rm_events_by_depth:
                        rm_events_by_depth[event.depth] = []
                    rm_events_by_depth[event.depth].append(event)

            # Compute RM stall times (waiting for LM at same depth)
            for depth, rm_events in rm_events_by_depth.items():
                lm_events = lm_events_by_depth.get(depth, [])
                if not lm_events:
                    continue

                # Find the latest LM completion at this depth
                latest_lm_end = max(e.end_time for e in lm_events)

                for rm_event in rm_events:
                    if rm_event.start_time > latest_lm_end:
                        rm_event.stall_time = rm_event.start_time - latest_lm_end

                    # Link dependency
                    latest_lm = max(lm_events, key=lambda e: e.end_time)
                    rm_event.triggered_by = latest_lm.event_id
                    if rm_event.event_id not in latest_lm.triggers:
                        latest_lm.triggers.append(rm_event.event_id)

            # Compute LM stall times (waiting for RM from previous depth)
            depths = sorted(set(lm_events_by_depth.keys()))
            for i, depth in enumerate(depths):
                if i == 0:
                    continue  # First depth has no previous RM to wait for

                prev_depth = depths[i - 1]
                prev_rm_events = rm_events_by_depth.get(prev_depth, [])
                if not prev_rm_events:
                    continue

                latest_prev_rm_end = max(e.end_time for e in prev_rm_events)

                for lm_event in lm_events_by_depth[depth]:
                    if lm_event.start_time > latest_prev_rm_end:
                        lm_event.stall_time = lm_event.start_time - latest_prev_rm_end

                    # Link dependency
                    latest_prev_rm = max(prev_rm_events, key=lambda e: e.end_time)
                    lm_event.triggered_by = latest_prev_rm.event_id
                    if lm_event.event_id not in latest_prev_rm.triggers:
                        latest_prev_rm.triggers.append(lm_event.event_id)

    def get_events(self) -> List[ExecutionEvent]:
        """Get all recorded events."""
        with self._events_lock:
            return list(self._events)

    def get_summary(self) -> Dict:
        """Get summary statistics of the trace.

        Returns:
            Dictionary containing timing and count statistics
        """
        with self._events_lock:
            if not self._events:
                return {}

            total_time = self._trace_end_time - self._trace_start_time

            lm_events = [e for e in self._events if e.event_type == EventType.LM_CALL]
            rm_events = [e for e in self._events if e.event_type == EventType.RM_CALL]
            tree_events = [e for e in self._events if e.event_type in (EventType.TREE_SELECT, EventType.TREE_EXPAND)]

            lm_total_time = sum(e.duration for e in lm_events)
            rm_total_time = sum(e.duration for e in rm_events)
            tree_total_time = sum(e.duration for e in tree_events)

            total_compute_time = lm_total_time + rm_total_time + tree_total_time
            total_stall_time = sum(e.stall_time for e in self._events)

            lm_stall_time = sum(e.stall_time for e in lm_events)
            rm_stall_time = sum(e.stall_time for e in rm_events)

            return {
                "total_execution_time_ms": total_time / 1000,
                "total_compute_time_ms": total_compute_time / 1000,
                "total_stall_time_ms": total_stall_time / 1000,
                "pipeline_efficiency": total_compute_time / total_time if total_time > 0 else 0,
                "language_model": {
                    "total_calls": len(lm_events),
                    "total_time_ms": lm_total_time / 1000,
                    "avg_call_duration_ms": lm_total_time / len(lm_events) / 1000 if lm_events else 0,
                    "stall_time_ms": lm_stall_time / 1000,
                },
                "reward_model": {
                    "total_calls": len(rm_events),
                    "total_time_ms": rm_total_time / 1000,
                    "avg_call_duration_ms": rm_total_time / len(rm_events) / 1000 if rm_events else 0,
                    "stall_time_ms": rm_stall_time / 1000,
                },
                "tree_operations": {
                    "total_calls": len(tree_events),
                    "total_time_ms": tree_total_time / 1000,
                },
            }

    def print_summary(self) -> None:
        """Print a formatted summary of the execution trace."""
        summary = self.get_summary()
        if not summary:
            print("No trace data available.")
            return

        print("\n" + "=" * 60)
        print("EXECUTION GRAPH SUMMARY")
        print("=" * 60)

        print("\n[Timing]")
        print(f"  Total execution time:    {summary['total_execution_time_ms']:.2f} ms")
        print(f"  Total compute time:      {summary['total_compute_time_ms']:.2f} ms")
        print(f"  Total stall time:        {summary['total_stall_time_ms']:.2f} ms")
        print(f"  Pipeline efficiency:     {summary['pipeline_efficiency'] * 100:.1f}%")

        lm = summary["language_model"]
        print("\n[Language Model (Policy)]")
        print(f"  Total calls:             {lm['total_calls']}")
        print(f"  Total time:              {lm['total_time_ms']:.2f} ms")
        print(f"  Avg call duration:       {lm['avg_call_duration_ms']:.2f} ms")

        rm = summary["reward_model"]
        print("\n[Reward Model (Value)]")
        print(f"  Total calls:             {rm['total_calls']}")
        print(f"  Total time:              {rm['total_time_ms']:.2f} ms")
        print(f"  Avg call duration:       {rm['avg_call_duration_ms']:.2f} ms")

        print("\n[Stall Analysis]")
        print(f"  LM waiting for RM:       {lm['stall_time_ms']:.2f} ms")
        print(f"  RM waiting for LM:       {rm['stall_time_ms']:.2f} ms")

        print("=" * 60 + "\n")
