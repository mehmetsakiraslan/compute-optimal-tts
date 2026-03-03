"""Lazy Pruning Beam Search: Deterministic LM/RM pipelining.

LM expands freely for `prune_interval` depths, then blocks until PRM scores
from `prune_interval` depths ago are guaranteed ready. This gives full
PRM-guided pruning (no accuracy loss) while overlapping LM and RM on
separate GPUs (throughput gain).

Algorithm (prune_interval=2):
  d=0: LM expand. RM starts scoring d=0 nodes async.
  d=1: LM expand (no prune). RM finishes d=0, starts d=1.
  d=2: WAIT for RM(d=0). PRUNE with guaranteed scores. LM expand.
  d=3: LM expand (no prune).
  d=4: WAIT for RM(d=2). PRUNE. LM expand.
  ...
"""

import queue
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger

from envs.base_env import CoTEnv, NoLegalActionException
from reason.guided_search.tree import LanguageNode, Node, SearchTree
from reason.profiling.nvtx_utils import nvtx_range, NVTXColors
from reason.profiling.execution_tracer import ExecutionTracer

# Queue item: (node, question_str, prm_answer_str, depth)
LazyPRMQueueItem = Tuple[LanguageNode, str, str, int]


class LazyPruningSearchTree(SearchTree):
    """Search tree with deterministic lazy pruning.

    LM generates children at each depth (parallelized via ThreadPoolExecutor).
    PRM scores in a background thread. Pruning happens every `prune_interval`
    depths, blocking until the PRM scores from `prune_interval` depths ago
    are guaranteed complete.
    """

    def __init__(self, cfg, max_frontier_width: int = 4, children_per_node: int = 1,
                 stop_str: Optional[List[str]] = None,
                 prune_interval: int = 2) -> None:
        super().__init__(cfg)
        self.max_frontier_width = max_frontier_width
        self.children_per_node = children_per_node
        self._stop_str = stop_str or []
        self.prune_interval = prune_interval

        # PRM scores: maps id(node) -> float, protected by lock
        self._prm_scores: Dict[int, float] = {}
        self._prm_lock = threading.Lock()
        # Lock for _completion_tokens (modified from ThreadPoolExecutor workers)
        self._tokens_lock = threading.Lock()

        # Per-depth synchronization for deterministic pruning
        self._depth_scored_events: Dict[int, threading.Event] = defaultdict(threading.Event)
        self._depth_pending_counts: Dict[int, int] = defaultdict(int)
        self._depth_pending_lock = threading.Lock()

    def lazy_beam_search(
        self,
        simulate_env: CoTEnv,
        beam_size: int,
        max_step: int,
        reward_model_fn: Optional[Callable] = None,
    ) -> List[Dict]:
        """Lazy pruning beam search entry point."""
        with nvtx_range("lazy_beam_search_total", NVTXColors.BEAM_RED):
            return self._lazy_beam_search_impl(
                simulate_env, beam_size, max_step, reward_model_fn
            )

    def _lazy_beam_search_impl(
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

        # Enqueue root's children for PRM scoring at depth -1 (pre-loop)
        root_answer = simulate_env.answer  # ""
        root_prm_items = []
        for child_key, child_node in root.children.items():
            prm_answer = root_answer + child_node.last_action
            root_prm_items.append((child_node, simulate_env.question, prm_answer))
        self._enqueue_prm_items(root_prm_items, prm_queue, depth=-1)

        # Build initial frontier: all children of root
        frontier: List[Tuple[LanguageNode, CoTEnv]] = []
        for child_key, child_node in root.children.items():
            frontier.append((child_node, simulate_env.copy()))

        # Initial cap (no pruning yet, just width limit)
        if len(frontier) > self.max_frontier_width:
            frontier = frontier[:self.max_frontier_width]

        end_nodes: List[Tuple[LanguageNode, CoTEnv]] = []

        # Phase 3: Depth loop with lazy pruning
        for depth in range(max_step + 1):
            tracer.set_current_depth(depth + 1)
            with nvtx_range(f"lazy_depth_{depth}", NVTXColors.DEPTH_CYAN):
                if not frontier:
                    break

                # PRUNE at multiples of prune_interval (deterministic wait)
                if depth >= self.prune_interval and depth % self.prune_interval == 0:
                    score_depth = depth - self.prune_interval
                    logger.info(
                        f"Lazy prune: depth={depth}, waiting for PRM scores from depth={score_depth}"
                    )
                    self._depth_scored_events[score_depth].wait()
                    effective_width = beam_size - len(end_nodes)
                    frontier = self._prune_frontier(frontier, effective_width)
                    logger.info(
                        f"Lazy prune: depth={depth}, frontier after prune: {len(frontier)}"
                    )

                if not frontier:
                    break

                next_frontier: List[Tuple[LanguageNode, CoTEnv]] = []
                depth_prm_items: List[Tuple[LanguageNode, str, str]] = []

                # Expand all frontier nodes in parallel using ThreadPoolExecutor
                with nvtx_range(f"lazy_expand_depth_{depth}", NVTXColors.TREE_BLUE):
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

                                # Collect PRM items (without depth tag yet)
                                depth_prm_items.extend(prm_items)

                            except Exception as e:
                                logger.warning(f"Error expanding node at depth {depth}: {e}")
                                node.set_as_terminate_node()
                                end_nodes.append((node, env))

                # Enqueue all PRM items for this depth
                self._enqueue_prm_items(depth_prm_items, prm_queue, depth=depth)

                # Cap frontier to beam_size to match sync's LM workload.
                # The PRM overlap only yields a net speedup when LM work
                # per depth equals sync; using max_frontier_width here
                # would expand more nodes than sync at every non-prune depth.
                frontier = self._cap_frontier_lightweight(
                    next_frontier, beam_size, len(end_nodes)
                )

        # Phase 4: Signal PRM done and wait
        prm_done_event.set()
        prm_thread.join(timeout=60)

        logger.info(
            f"Lazy search complete: {len(end_nodes)} end_nodes, "
            f"{len(frontier)} remaining frontier, "
            f"{len(self._prm_scores)} PRM scores stored"
        )

        # Phase 5: Final selection using PRM scores
        return self._select_best_trajectories(
            end_nodes, frontier, beam_size, api_call_completion_tokens
        )

    # ------------------------------------------------------------------
    # PRM enqueue helper
    # ------------------------------------------------------------------

    def _enqueue_prm_items(
        self,
        prm_items: List[Tuple[LanguageNode, str, str]],
        prm_queue: queue.Queue,
        depth: int,
    ) -> None:
        """Enqueue PRM items tagged with their depth for synchronization."""
        if not prm_items:
            # No items at this depth — pre-set the event so wait() won't deadlock
            self._depth_scored_events[depth].set()
            return

        with self._depth_pending_lock:
            self._depth_pending_counts[depth] += len(prm_items)

        for node, question, prm_answer in prm_items:
            prm_queue.put((node, question, prm_answer, depth))

    # ------------------------------------------------------------------
    # PRM background worker
    # ------------------------------------------------------------------

    def _prm_worker(
        self,
        prm_queue: queue.Queue,
        done_event: threading.Event,
        reward_model_fn: Optional[Callable],
    ) -> None:
        """Background PRM worker thread.

        Batches nodes from the queue and scores them via rm_call.
        Stores scores in self._prm_scores protected by self._prm_lock.
        After scoring, decrements per-depth pending counts and signals
        completion events.
        """
        BATCH_SIZE = 32
        POLL_TIMEOUT = 0.1  # seconds

        def _score_batch(batch: List[LazyPRMQueueItem]) -> None:
            if not batch or reward_model_fn is None:
                # Still need to decrement pending counts
                if batch:
                    self._decrement_depth_counts(batch)
                return

            prm_inputs = []
            batch_nodes = []
            batch_depths = []
            skipped_depths = []
            for node, question, prm_answer, depth in batch:
                if node.is_root() or node.last_action is None:
                    skipped_depths.append(depth)
                    continue
                prm_inputs.append((question, prm_answer))
                batch_nodes.append(node)
                batch_depths.append(depth)

            # Decrement counts for skipped items
            if skipped_depths:
                self._decrement_depth_counts_by_depth(skipped_depths)

            if not prm_inputs:
                return

            try:
                with nvtx_range("lazy_prm_batch", NVTXColors.RM_YELLOW):
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

            # Decrement pending counts for scored items
            self._decrement_depth_counts_by_depth(batch_depths)

        while True:
            batch: List[LazyPRMQueueItem] = []

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

    def _decrement_depth_counts(self, batch: List[LazyPRMQueueItem]) -> None:
        """Decrement pending counts for a batch of items, signaling events when done."""
        depth_counts: Dict[int, int] = defaultdict(int)
        for _, _, _, depth in batch:
            depth_counts[depth] += 1

        with self._depth_pending_lock:
            for depth, count in depth_counts.items():
                self._depth_pending_counts[depth] -= count
                if self._depth_pending_counts[depth] <= 0:
                    self._depth_scored_events[depth].set()

    def _decrement_depth_counts_by_depth(self, depths: List[int]) -> None:
        """Decrement pending counts for a list of depths, signaling events when done."""
        depth_counts: Dict[int, int] = defaultdict(int)
        for depth in depths:
            depth_counts[depth] += 1

        with self._depth_pending_lock:
            for depth, count in depth_counts.items():
                self._depth_pending_counts[depth] -= count
                if self._depth_pending_counts[depth] <= 0:
                    self._depth_scored_events[depth].set()

    # ------------------------------------------------------------------
    # Frontier pruning (deterministic — scores guaranteed available)
    # ------------------------------------------------------------------

    def _prune_frontier(
        self,
        frontier: List[Tuple[LanguageNode, CoTEnv]],
        effective_width: int,
    ) -> List[Tuple[LanguageNode, CoTEnv]]:
        """Prune frontier using PRM-guided tiered scoring.

        Unlike _cap_frontier in AsyncSearchTree, this does NOT poll/wait —
        scores are guaranteed available from the depth event wait() above.
        """
        cap = min(self.max_frontier_width, effective_width) if effective_width > 0 else self.max_frontier_width

        if cap <= 0:
            return []

        if len(frontier) <= cap:
            return frontier

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

        scored = sorted(frontier, key=score_fn, reverse=True)
        kept = scored[:cap]
        pruned_count = len(frontier) - len(kept)
        if pruned_count > 0:
            tiers = [score_fn(e)[0] for e in kept]
            logger.info(
                f"Lazy prune: {len(frontier)} -> {len(kept)} "
                f"(cap={cap}, tiers={tiers})"
            )
        return kept

    def _cap_frontier_lightweight(
        self,
        frontier: List[Tuple[LanguageNode, CoTEnv]],
        beam_size: int,
        num_end_nodes: int,
    ) -> List[Tuple[LanguageNode, CoTEnv]]:
        """Cap frontier to beam_size at every depth.

        Matches sync beam_search's frontier width so the LM workload per
        depth is identical. The PRM overlap then yields a pure speed gain
        (no extra LM cost). Uses PRM scores opportunistically if the
        background worker has already scored the node or its parent,
        falls back to LM log-prob.
        """
        cap = max(0, beam_size - num_end_nodes)

        if cap <= 0:
            return []

        if len(frontier) <= cap:
            return frontier

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

        scored = sorted(frontier, key=score_fn, reverse=True)
        kept = scored[:cap]
        tiers = [score_fn(e)[0] for e in kept]
        logger.info(
            f"Lightweight cap: {len(frontier)} -> {len(kept)} "
            f"(beam_size={beam_size}, end_nodes={num_end_nodes}, tiers={tiers})"
        )
        return kept

    # ------------------------------------------------------------------
    # Node expansion (reused from AsyncSearchTree)
    # ------------------------------------------------------------------

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
    ) -> Tuple[List[Tuple[LanguageNode, CoTEnv]], int, List[Tuple[LanguageNode, str, str]]]:
        """Expand a single frontier node: step env, then generate children via LM.

        Returns:
            (list of (child_node, child_env), api_completion_tokens,
             list of (node, question, prm_answer) 3-tuples — depth tag added by caller)
        """
        tokens_used = 0
        children_list: List[Tuple[LanguageNode, CoTEnv]] = []
        prm_items: List[Tuple[LanguageNode, str, str]] = []

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
        parent_answer = env.answer
        question = env.question
        for child_key, child_node in node.children.items():
            child_env = env.copy()
            children_list.append((child_node, child_env))
            prm_items.append((child_node, question, parent_answer + child_node.last_action))

        return children_list, tokens_used, prm_items

    # ------------------------------------------------------------------
    # Final trajectory selection (reused from AsyncSearchTree)
    # ------------------------------------------------------------------

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

        # Score completed trajectories
        scored_end_complete = []  # contain stop_str -> true conclusions
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
