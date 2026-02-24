"""Async Beam Search: Non-blocking LM/PRM pipeline.

LM expands all frontier nodes at each depth without waiting for PRM.
PRM scores nodes in a background thread. PRM scores are used for final
trajectory selection (and optionally for frontier capping when available).
"""

import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

from envs.base_env import CoTEnv, NoLegalActionException
from reason.guided_search.tree import LanguageNode, Node, SearchTree
from reason.profiling.nvtx_utils import nvtx_range, NVTXColors
from reason.profiling.execution_tracer import ExecutionTracer

# Queue item: (node, question_str, prm_answer_str)
PRMQueueItem = Tuple[LanguageNode, str, str]


class AsyncSearchTree(SearchTree):
    """Search tree with async PRM scoring.

    LM generates children at each depth synchronously (parallelized across
    frontier nodes with ThreadPoolExecutor). PRM scores are computed in a
    dedicated background thread consuming from a queue.
    """

    def __init__(self, cfg, max_frontier_width: int = 4, children_per_node: int = 1,
                 stop_str: Optional[List[str]] = None,
                 prm_wait_timeout: float = 0.5,
                 prm_coverage_threshold: float = 0.8) -> None:
        super().__init__(cfg)
        self.max_frontier_width = max_frontier_width
        self.children_per_node = children_per_node
        self._stop_str = stop_str or []
        self.prm_wait_timeout = prm_wait_timeout
        self.prm_coverage_threshold = prm_coverage_threshold
        # PRM scores: maps id(node) -> float, protected by lock
        self._prm_scores: Dict[int, float] = {}
        self._prm_lock = threading.Lock()
        # Lock for _completion_tokens (modified from ThreadPoolExecutor workers)
        self._tokens_lock = threading.Lock()

    def async_beam_search(
        self,
        simulate_env: CoTEnv,
        beam_size: int,
        max_step: int,
        reward_model_fn: Optional[Callable] = None,
    ) -> List[Dict]:
        """Async beam search entry point."""
        with nvtx_range("async_beam_search_total", NVTXColors.BEAM_RED):
            return self._async_beam_search_impl(
                simulate_env, beam_size, max_step, reward_model_fn
            )

    def _async_beam_search_impl(
        self,
        simulate_env: CoTEnv,
        beam_size: int,
        max_step: int,
        reward_model_fn: Optional[Callable] = None,
    ) -> List[Dict]:
        api_call_completion_tokens = 0
        tracer = ExecutionTracer.get_instance()
        tracer.set_current_depth(0)

        # Phase 1: Init — reset env, create root, LM generates initial children
        _, info = simulate_env.reset(update_legal_action=True)
        api_call_completion_tokens += info["api_completion_token"]

        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state(model_name='raw'))
            self._expand_leaf_node_no_prm(root, simulate_env)
            self.root = root

        # Phase 2: Start PRM worker thread
        prm_queue: queue.Queue = queue.Queue()
        prm_done_event = threading.Event()
        prm_thread = threading.Thread(
            target=self._prm_worker,
            args=(prm_queue, prm_done_event, reward_model_fn),
            daemon=True,
        )
        prm_thread.start()

        # Enqueue root's children for PRM scoring
        # At root: simulate_env.answer = "", so PRM answer = "" + child.last_action
        root_answer = simulate_env.answer  # ""
        for child_key, child_node in root.children.items():
            prm_answer = root_answer + child_node.last_action
            prm_queue.put((child_node, simulate_env.question, prm_answer))

        # Build initial frontier: all children of root
        # Frontier entries: (node, env_copy)
        frontier: List[Tuple[LanguageNode, CoTEnv]] = []
        for child_key, child_node in root.children.items():
            frontier.append((child_node, simulate_env.copy()))

        # Cap frontier to max width
        frontier = self._cap_frontier(frontier, beam_size=beam_size)

        end_nodes: List[Tuple[LanguageNode, CoTEnv]] = []

        # Phase 3: Depth loop — expand without waiting for PRM
        for depth in range(max_step + 1):
            tracer.set_current_depth(depth + 1)
            with nvtx_range(f"async_depth_{depth}", NVTXColors.DEPTH_CYAN):
                if not frontier:
                    break

                next_frontier: List[Tuple[LanguageNode, CoTEnv]] = []

                # Expand all frontier nodes in parallel using ThreadPoolExecutor
                with nvtx_range(f"async_expand_depth_{depth}", NVTXColors.TREE_BLUE):
                    with ThreadPoolExecutor(max_workers=min(4, len(frontier))) as executor:
                        futures = {
                            executor.submit(
                                self._expand_single_node, node, env
                            ): (node, env)
                            for node, env in frontier
                        }
                        for future in as_completed(futures):
                            node, env = futures[future]
                            try:
                                result = future.result()
                                children_list, tokens_used, prm_items = result
                                api_call_completion_tokens += tokens_used

                                for child_node, child_env in children_list:
                                    if child_node.terminated:
                                        end_nodes.append((child_node, child_env))
                                    else:
                                        next_frontier.append((child_node, child_env))

                                # Enqueue pre-computed PRM items
                                for prm_item in prm_items:
                                    prm_queue.put(prm_item)

                            except Exception as e:
                                logger.warning(f"Error expanding node at depth {depth}: {e}")
                                node.set_as_terminate_node()
                                end_nodes.append((node, env))

                # Cap frontier with beam_size shrunk by completed trajectories
                frontier = self._cap_frontier(next_frontier, beam_size=beam_size - len(end_nodes))

                # Don't exit early based on beam_size — we want to
                # explore all depths and let PRM pick the best completed
                # trajectory at the end. With multiple frontier nodes
                # expanded in parallel, premature terminals are common
                # but often low-quality.

        # Phase 4: Signal PRM done and wait
        prm_done_event.set()
        prm_thread.join(timeout=60)

        logger.info(
            f"Async search complete: {len(end_nodes)} end_nodes, "
            f"{len(frontier)} remaining frontier, "
            f"{len(self._prm_scores)} PRM scores stored"
        )

        # Phase 5: Final selection using PRM scores
        return self._select_best_trajectories(
            end_nodes, frontier, beam_size, api_call_completion_tokens
        )

    def _expand_leaf_node_no_prm(
        self,
        node: Node,
        simulate_env: CoTEnv,
    ) -> None:
        """Expand leaf node with LM only (no PRM call).

        Creates LanguageNode children with initial_value=0.0.
        Mirrors _expand_leaf_node but skips the rm_call.
        """
        with nvtx_range("expand_leaf_no_prm", NVTXColors.TREE_BLUE):
            text_state = simulate_env.get_state(model_name='raw')
            leaf_value = node._initial_value

            assert len(simulate_env.legal_actions) > 0
            assert len(node.children) == 0

            for i, action_dict in enumerate(simulate_env.legal_actions):
                action, prob = action_dict["action"], action_dict["prob"]
                model_name = action_dict["model_name"]

                if self.direct_io:
                    node.children[i] = LanguageNode(
                        parent=node,
                        prior_p=prob,
                        text_state=text_state,
                        last_action=action,
                        initial_value=0.0,
                        parent_value=leaf_value,
                        num_generated_token=action_dict["num_token"],
                        model_name=model_name,
                    )
                else:
                    node.children[action] = LanguageNode(
                        parent=node,
                        prior_p=prob,
                        text_state=text_state,
                        last_action=action,
                        initial_value=0.0,
                        parent_value=leaf_value,
                        num_generated_token=action_dict["num_token"],
                        model_name=model_name,
                    )

                # Set terminal node
                if simulate_env._next_state_terminated[action]:
                    if self.direct_io:
                        node.children[i].set_as_terminate_node()
                    else:
                        node.children[action].set_as_terminate_node()

            if len(node.children) == 0:
                logger.warning(f"Prune all current children at node {node.last_action}")

            # Collect num tokens (thread-safe)
            if not node.has_collected_token_num:
                token_sum = sum(c.num_generated_token for c in node.children.values())
                with self._tokens_lock:
                    self._completion_tokens += token_sum
                node.has_collected_token_num = True

    def _expand_single_node(
        self,
        node: LanguageNode,
        env: CoTEnv,
    ) -> Tuple[List[Tuple[LanguageNode, CoTEnv]], int, List[PRMQueueItem]]:
        """Expand a single frontier node: step env, then generate children via LM.

        Returns:
            (list of (child_node, child_env), api_completion_tokens, list of PRM queue items)
        """
        tokens_used = 0
        children_list: List[Tuple[LanguageNode, CoTEnv]] = []
        prm_items: List[PRMQueueItem] = []

        # Step the environment with this node's action
        _, _, terminated, truncated, info = env.step(
            node.last_action,
            update_legal_action=(self.direct_io == 0),
            model_name=node.model_name,
            reward=node._initial_value,
            num_token=node.num_generated_token,
            prob=node.prior_p,
        )
        tokens_used += info["api_completion_token"]

        if terminated or truncated:
            # Check if answer contains stop_str (e.g. \boxed{})
            has_stop = any(s in env.answer for s in self._stop_str) if self._stop_str else False
            if has_stop or truncated:
                # True terminal — answer is complete
                node.set_as_terminate_node()
                prm_items.append((node, env.question, env.answer))
                return [(node, env)], tokens_used, prm_items
            else:
                # Premature termination: step ended (no \n\n) but no \boxed{} yet.
                # Force LM to generate more steps from current state.
                try:
                    env._legal_actions, extra_tokens = env.update_legal_actions()
                    tokens_used += extra_tokens
                    logger.debug(
                        f"Force-continued premature terminal at depth "
                        f"{len(env.action_history)}, got {len(env._legal_actions)} candidates"
                    )
                    # Fall through to expansion below
                except (NoLegalActionException, Exception):
                    # Cannot generate more — treat as truly terminal
                    node.set_as_terminate_node()
                    prm_items.append((node, env.question, env.answer))
                    return [(node, env)], tokens_used, prm_items

        # Expand: generate children via LM (no PRM)
        try:
            self._expand_leaf_node_no_prm(node, env)
        except (NoLegalActionException, Exception) as e:
            logger.warning(f"Failed to expand node: {e}")
            node.set_as_terminate_node()
            prm_items.append((node, env.question, env.answer))
            return [(node, env)], tokens_used, prm_items

        # Build child entries
        # env.answer = answer up to (and including) this node's action
        # child.last_action = the new candidate action
        # PRM answer = env.answer + child.last_action
        parent_answer = env.answer
        question = env.question
        for child_key, child_node in node.children.items():
            child_env = env.copy()
            children_list.append((child_node, child_env))
            prm_items.append((child_node, question, parent_answer + child_node.last_action))

        return children_list, tokens_used, prm_items

    def _prm_worker(
        self,
        prm_queue: queue.Queue,
        done_event: threading.Event,
        reward_model_fn: Optional[Callable],
    ) -> None:
        """Background PRM worker thread.

        Batches nodes from the queue and scores them via rm_call.
        Stores scores in self._prm_scores protected by self._prm_lock.
        """
        BATCH_SIZE = 32
        POLL_TIMEOUT = 0.1  # seconds

        def _score_batch(batch: List[PRMQueueItem]) -> None:
            if not batch or reward_model_fn is None:
                return

            prm_inputs = []
            batch_nodes = []
            for node, question, prm_answer in batch:
                if node.is_root() or node.last_action is None:
                    continue
                prm_inputs.append((question, prm_answer))
                batch_nodes.append(node)

            if not prm_inputs:
                return

            try:
                with nvtx_range("async_prm_batch", NVTXColors.RM_YELLOW):
                    prm_results = reward_model_fn(prm_inputs)

                with self._prm_lock:
                    for node, rs in zip(batch_nodes, prm_results):
                        if isinstance(rs, list) and len(rs) > 0:
                            score = rs[-1]  # prm-last
                        elif isinstance(rs, (int, float)):
                            score = float(rs)
                        else:
                            score = 0.0
                        self._prm_scores[id(node)] = score
                        node._initial_value = score
                logger.info(
                    f"PRM scored {len(batch_nodes)} nodes, "
                    f"scores: {[self._prm_scores[id(n)] for n in batch_nodes]}"
                )
            except Exception as e:
                logger.warning(f"PRM batch scoring failed: {e}")
                import traceback
                traceback.print_exc()
                with self._prm_lock:
                    for node in batch_nodes:
                        self._prm_scores[id(node)] = 0.0

        while True:
            batch: List[PRMQueueItem] = []

            # Collect up to BATCH_SIZE items from queue
            try:
                item = prm_queue.get(timeout=POLL_TIMEOUT)
                batch.append(item)
            except queue.Empty:
                if done_event.is_set() and prm_queue.empty():
                    break
                continue

            # Drain more items without blocking
            while len(batch) < BATCH_SIZE:
                try:
                    item = prm_queue.get_nowait()
                    batch.append(item)
                except queue.Empty:
                    break

            _score_batch(batch)

        # Drain remaining items after done signal
        while not prm_queue.empty():
            batch = []
            while not prm_queue.empty() and len(batch) < BATCH_SIZE:
                try:
                    batch.append(prm_queue.get_nowait())
                except queue.Empty:
                    break
            _score_batch(batch)

    def _wait_for_prm_coverage(
        self,
        frontier: List[Tuple[LanguageNode, CoTEnv]],
    ) -> None:
        """Poll for PRM scores until coverage threshold is met or timeout expires.

        At early depths where PRM hasn't scored anything, this times out quickly.
        At later depths, PRM has caught up and coverage is met immediately.
        """
        if not frontier:
            return

        deadline = time.monotonic() + self.prm_wait_timeout
        required = int(len(frontier) * self.prm_coverage_threshold)

        while time.monotonic() < deadline:
            covered = 0
            with self._prm_lock:
                for node, _ in frontier:
                    if id(node) in self._prm_scores:
                        covered += 1
                    elif node.parent is not None and id(node.parent) in self._prm_scores:
                        covered += 1
            if covered >= required:
                return
            time.sleep(0.05)

    def _cap_frontier(
        self,
        frontier: List[Tuple[LanguageNode, CoTEnv]],
        beam_size: int = 0,
    ) -> List[Tuple[LanguageNode, CoTEnv]]:
        """Cap frontier using PRM-guided tiered scoring.

        Effective width = min(max_frontier_width, beam_size) when beam_size > 0.
        Waits briefly for PRM scores, then ranks nodes using:
          tier 2: node's own PRM score (best signal)
          tier 1: parent's PRM score (inherited)
          tier 0: LM log-prob (fallback for early depths)
        """
        effective_width = self.max_frontier_width
        if beam_size > 0:
            effective_width = min(effective_width, beam_size)

        if effective_width <= 0:
            return []

        if len(frontier) <= effective_width:
            return frontier

        # Wait briefly for PRM scores to arrive
        self._wait_for_prm_coverage(frontier)

        def score_fn(entry: Tuple[LanguageNode, CoTEnv]) -> Tuple[int, float]:
            node = entry[0]
            with self._prm_lock:
                node_score = self._prm_scores.get(id(node))
                if node_score is not None:
                    return (2, node_score)
                if node.parent is not None:
                    parent_score = self._prm_scores.get(id(node.parent))
                    if parent_score is not None:
                        return (1, parent_score)
            return (0, node.prior_p)

        # Sort descending by (tier, score), take top effective_width
        scored = sorted(frontier, key=score_fn, reverse=True)
        kept = scored[:effective_width]
        pruned_count = len(frontier) - len(kept)
        if pruned_count > 0:
            tiers = [score_fn(e)[0] for e in kept]
            logger.info(
                f"Frontier capped: {len(frontier)} -> {len(kept)} "
                f"(beam_size={beam_size}, tiers={tiers})"
            )
        return kept

    def _select_best_trajectories(
        self,
        end_nodes: List[Tuple[LanguageNode, CoTEnv]],
        remaining_frontier: List[Tuple[LanguageNode, CoTEnv]],
        beam_size: int,
        api_call_completion_tokens: int,
    ) -> List[Dict]:
        """Select best trajectories using PRM scores.

        Called after PRM thread has joined, so all scores are final.
        Prefers completed (terminated) trajectories over incomplete ones.
        """
        # Ensure all end_nodes have their env stepped with the node's action.
        # Nodes returned directly from _expand_single_node (Path A) are already
        # stepped, but children marked terminal by _expand_leaf_node_no_prm
        # (Path B) have envs copied from the parent — not yet stepped.
        for node, env in end_nodes:
            if not env.action_history or env.action_history[-1] != node.last_action:
                try:
                    env.step(
                        node.last_action,
                        update_legal_action=False,
                        model_name=node.model_name,
                        reward=node._initial_value,
                        num_token=node.num_generated_token,
                        prob=node.prior_p,
                    )
                except Exception:
                    pass

        # Score completed trajectories, preferring those with stop_str (e.g. \boxed)
        # which indicates a fully-concluded answer
        scored_end_complete = []  # contain stop_str → true conclusions
        scored_end_partial = []   # terminated but no stop_str
        for node, env in end_nodes:
            score = self._prm_scores.get(id(node), 0.0)
            answer_text = env.answer
            has_stop = any(s in answer_text for s in self._stop_str) if self._stop_str else False
            if has_stop:
                scored_end_complete.append((score, node, env))
            else:
                scored_end_partial.append((score, node, env))
        scored_end_complete.sort(key=lambda x: x[0], reverse=True)
        scored_end_partial.sort(key=lambda x: x[0], reverse=True)

        logger.info(
            f"Trajectory selection: {len(scored_end_complete)} complete (with stop_str), "
            f"{len(scored_end_partial)} partial, "
            f"complete scores: {[round(s, 4) for s, _, _ in scored_end_complete[:5]]}, "
            f"partial scores: {[round(s, 4) for s, _, _ in scored_end_partial[:5]]}"
        )
        if scored_end_partial and not scored_end_complete:
            logger.warning(
                f"No complete trajectories found (stop_str={self._stop_str}). "
                f"All {len(scored_end_partial)} end_nodes are partial."
            )

        # Prefer trajectories with stop_str, fall back to partial
        scored_end = scored_end_complete + scored_end_partial

        # If we have enough completed trajectories, use those
        if len(scored_end) >= beam_size:
            selected = scored_end[:beam_size]
        else:
            # Not enough completed — supplement with frontier nodes
            # For frontier nodes, step their env so env.answer is complete
            scored_frontier = []
            for node, env in remaining_frontier:
                stepped_env = env.copy()
                try:
                    stepped_env.step(
                        node.last_action,
                        update_legal_action=False,
                        model_name=node.model_name,
                        reward=node._initial_value,
                        num_token=node.num_generated_token,
                        prob=node.prior_p,
                    )
                except Exception:
                    continue
                score = self._prm_scores.get(id(node), 0.0)
                scored_frontier.append((score, node, stepped_env))
            scored_frontier.sort(key=lambda x: x[0], reverse=True)

            selected = scored_end + scored_frontier
            selected.sort(key=lambda x: x[0], reverse=True)
            selected = selected[:beam_size]

        if not selected:
            return []

        # Build trajectory list in same format as standard beam_search
        traj_list = []
        for i, (score, node, env) in enumerate(selected):
            parent_score = 0.0
            if node.parent is not None:
                parent_score = self._prm_scores.get(id(node.parent), 0.0)

            traj_list.append({
                "path_idx": i,
                "text": env.answer,
                "value": score,
                "parent_value": parent_score,
                "q_plus_a": score,
                "api_completion_tokens": 0,
                "tree_completion_tokens": 0,
                "reward_history": env.reward_history,
                "token_history": env.token_history,
                "prob_history": env.prob_history,
                "model_history": env.model_history,
            })

        if traj_list:
            traj_list[-1]["tree_completion_tokens"] = self._completion_tokens
            traj_list[-1]["api_completion_tokens"] = api_call_completion_tokens

        return traj_list
