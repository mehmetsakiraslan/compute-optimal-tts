"""Traced caller wrappers for execution graph profiling.

This module provides wrapper classes around the LM and RM callers that
automatically instrument calls with execution tracing events.
"""

from functools import partial
from typing import List, Optional, Tuple, Union

from reason.inference.lm_call import (
    VLLMRemoteCaller,
    LMCallingConfig,
    ConcatedLMGenResult,
)
from reason.inference.rm_call import (
    RMRemoteCaller,
    RemoteRewardModelConfig,
)
from reason.profiling.execution_tracer import (
    ExecutionTracer,
    EventType,
)


class TracedVLLMRemoteCaller(VLLMRemoteCaller):
    """Wrapper around VLLMRemoteCaller that adds execution tracing.

    This class inherits from VLLMRemoteCaller and wraps the __call__ method
    to record LM_CALL events in the execution tracer.
    """

    def __init__(
        self,
        model_name,
        model_path,
        controller_addr="http://0.0.0.0:28777",
        llm_step_tag: str = None,
        apply_chat_template: bool = False,
        multi_gpu: bool = False,
        serve_type: str = "fastchat",
        double_line_break: int = 0,
        tracer: Optional[ExecutionTracer] = None,
    ):
        super().__init__(
            model_name=model_name,
            model_path=model_path,
            controller_addr=controller_addr,
            llm_step_tag=llm_step_tag,
            apply_chat_template=apply_chat_template,
            multi_gpu=multi_gpu,
            serve_type=serve_type,
            double_line_break=double_line_break,
        )
        self._tracer = tracer or ExecutionTracer.get_instance()

    def __call__(self, messages: str, config: LMCallingConfig) -> ConcatedLMGenResult:
        """Call the LM with tracing instrumentation.

        Args:
            messages: Input messages/prompt
            config: LM calling configuration

        Returns:
            ConcatedLMGenResult with generation results
        """
        # Estimate input tokens (rough approximation)
        input_tokens = len(messages) // 4 if isinstance(messages, str) else 0

        # Start the trace event
        event = self._tracer.start_event(
            event_type=EventType.LM_CALL,
            model="Policy",
            metadata={
                "input_tokens_estimate": input_tokens,
                "n": config.n,
                "temperature": config.temperature,
                "max_new_tokens": config.max_new_tokens,
                "model_name": self.model_name,
            },
        )

        try:
            # Call the parent implementation
            result = super().__call__(messages, config)

            # Update metadata with actual output info
            self._tracer.end_event(
                event,
                metadata_update={
                    "output_tokens": result.completion_tokens,
                    "num_sequences": len(result.text),
                    "finish_reasons": result.finish_reason,
                },
            )

            return result

        except Exception as e:
            self._tracer.end_event(
                event,
                metadata_update={"error": str(e)},
            )
            raise


class TracedRMRemoteCaller(RMRemoteCaller):
    """Wrapper around RMRemoteCaller that adds execution tracing.

    This class inherits from RMRemoteCaller and wraps the __call__ method
    to record RM_CALL events in the execution tracer.
    """

    def __init__(
        self,
        config: RemoteRewardModelConfig,
        tokenizer,
        tracer: Optional[ExecutionTracer] = None,
    ):
        super().__init__(config=config, tokenizer=tokenizer)
        self._tracer = tracer or ExecutionTracer.get_instance()

    def __call__(
        self,
        qa_pairs: Union[Tuple[str, str], List[Tuple[str, str]]],
        model_names: List[str],
        verbose: Optional[bool] = False,
        local: Optional[bool] = False,
        legal_action: Optional[List[str]] = [],
        process: Optional[bool] = True,
        timeout: Optional[int] = 0,
    ) -> Union[List[int], List[List[int]]]:
        """Call the RM with tracing instrumentation.

        Args:
            qa_pairs: Question-answer pairs to score
            model_names: Model names for processing
            verbose: Enable verbose output
            local: Use local inference
            legal_action: Legal actions for debugging
            process: Whether to process input
            timeout: Request timeout

        Returns:
            Reward scores for each input
        """
        # Calculate number of candidates
        if isinstance(qa_pairs, tuple) and isinstance(qa_pairs[0], str):
            num_candidates = 1
        else:
            num_candidates = len(qa_pairs)

        # Start the trace event
        event = self._tracer.start_event(
            event_type=EventType.RM_CALL,
            model="Value",
            metadata={
                "num_candidates": num_candidates,
                "model_name": self.model_name,
                "local": local,
            },
        )

        try:
            # Call the parent implementation
            result = super().__call__(
                qa_pairs=qa_pairs,
                model_names=model_names,
                verbose=verbose,
                local=local,
                legal_action=legal_action,
                process=process,
                timeout=timeout,
            )

            # Update metadata with result info
            if isinstance(result, list):
                num_scores = len(result)
            else:
                num_scores = 1

            self._tracer.end_event(
                event,
                metadata_update={
                    "num_scores_returned": num_scores,
                },
            )

            return result

        except Exception as e:
            self._tracer.end_event(
                event,
                metadata_update={"error": str(e)},
            )
            raise


def create_traced_rm_call(rm_call: RMRemoteCaller, model_names: List[str]) -> callable:
    """Create a traced partial RM call function.

    This is a helper to create a traced version of the rm_call that is
    already bound with model_names, similar to how evaluate.py does it.

    Args:
        rm_call: The traced RM caller instance
        model_names: Model names to bind

    Returns:
        A partial function that can be called with just qa_pairs
    """
    return partial(rm_call, model_names=model_names)


def wrap_existing_rm_call(rm_call_partial) -> callable:
    """Wrap an existing partial RM call with tracing.

    This is useful when you already have a configured rm_call partial
    and want to add tracing to it.

    Args:
        rm_call_partial: An existing partial RM call

    Returns:
        A wrapped callable that traces RM calls
    """
    tracer = ExecutionTracer.get_instance()

    def traced_rm_call(*args, **kwargs):
        # Determine number of candidates
        qa_pairs = args[0] if args else kwargs.get("qa_pairs", kwargs.get("prm_inputs", []))
        if isinstance(qa_pairs, tuple) and isinstance(qa_pairs[0], str):
            num_candidates = 1
        elif isinstance(qa_pairs, list):
            num_candidates = len(qa_pairs)
        else:
            num_candidates = 0

        event = tracer.start_event(
            event_type=EventType.RM_CALL,
            model="Value",
            metadata={"num_candidates": num_candidates},
        )

        try:
            result = rm_call_partial(*args, **kwargs)
            tracer.end_event(event)
            return result
        except Exception as e:
            tracer.end_event(event, metadata_update={"error": str(e)})
            raise

    return traced_rm_call


def wrap_existing_lm_call(lm_call: VLLMRemoteCaller) -> VLLMRemoteCaller:
    """Wrap an existing LM call with tracing by replacing its __call__ method.

    Args:
        lm_call: An existing VLLMRemoteCaller instance

    Returns:
        The same instance with __call__ wrapped for tracing
    """
    tracer = ExecutionTracer.get_instance()
    original_call = lm_call.__call__

    def traced_call(messages: str, config: LMCallingConfig) -> ConcatedLMGenResult:
        input_tokens = len(messages) // 4 if isinstance(messages, str) else 0

        event = tracer.start_event(
            event_type=EventType.LM_CALL,
            model="Policy",
            metadata={
                "input_tokens_estimate": input_tokens,
                "n": config.n,
                "temperature": config.temperature,
                "max_new_tokens": config.max_new_tokens,
                "model_name": lm_call.model_name,
            },
        )

        try:
            result = original_call(messages, config)
            tracer.end_event(
                event,
                metadata_update={
                    "output_tokens": result.completion_tokens,
                    "num_sequences": len(result.text),
                },
            )
            return result
        except Exception as e:
            tracer.end_event(event, metadata_update={"error": str(e)})
            raise

    # Replace the __call__ method
    import types
    lm_call.__call__ = types.MethodType(lambda self, m, c: traced_call(m, c), lm_call)

    return lm_call
