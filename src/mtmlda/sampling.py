import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Union

import anytree as atree
import numpy as np

from . import jobhandling, logging, mcmc, mltree
from .mltree import MLTreeSearchFunctions as mltree_search


# ==================================================================================================
@dataclass
class SamplerSetupSettings:
    num_levels: int
    subsampling_rates: list
    max_tree_height: int = 50
    underflow_threshold: float = -1000
    rng_seed_mltree: int = 0
    rng_seed_node_init: int = 0
    mltree_path: str = None


@dataclass
class SamplerRunSettings:
    num_samples: int
    initial_state: np.ndarray
    initial_node: mltree.MTNode = None
    num_threads: int = 1
    print_interval: int = 1
    tree_render_interval: int = 1


@dataclass
class RNGStates:
    proposal: np.random.Generator
    mltree: np.random.Generator
    node_init: np.random.Generator


# ==================================================================================================
class MTMLDASampler:
    def __init__(
        self,
        setup_settings: SamplerSetupSettings,
        logger_settings: logging.LoggerSettings,
        models: Sequence[Callable],
        accept_rate_estimator: mcmc.BaseAcceptRateEstimator,
        ground_proposal: mcmc.BaseProposal,
    ) -> None:
        self._models = models
        self._num_levels = setup_settings.num_levels
        self._subsampling_rates = setup_settings.subsampling_rates
        self._maximum_tree_height = setup_settings.max_tree_height
        self._underflow_threshold = setup_settings.underflow_threshold

        self._accept_rate_estimator = accept_rate_estimator
        self._ground_proposal = ground_proposal
        self._rng_node_init = np.random.default_rng(setup_settings.rng_seed_node_init)
        self._mcmc_kernel = mcmc.MLMetropolisHastingsKernel(ground_proposal)

        self._mltree_modifier = mltree.MLTreeModifier(
            setup_settings.num_levels,
            ground_proposal,
            setup_settings.subsampling_rates,
            setup_settings.rng_seed_mltree,
        )
        self._mltree_visualizer = mltree.MLTreeVisualizer(setup_settings.mltree_path)
        self._logger = logging.MTMLDALogger(logger_settings)
        self._run_statistics, self._debug_statistics = self._init_statistics()

        self._start_time = None
        self._num_samples = None
        self._print_interval = None
        self._tree_render_interval = None
        self._job_handler = None
        
        self._num_generated_samples = [0,] * self._num_levels
        self._num_accepted_samples = [0,] * self._num_levels

    # ----------------------------------------------------------------------------------------------
    def run(self, run_settings: SamplerRunSettings) -> list[np.ndarray]:
        self._start_time = time.time()
        self._num_samples = run_settings.num_samples
        self._print_interval = run_settings.print_interval
        self._tree_render_interval = run_settings.tree_render_interval
        num_threads = run_settings.num_threads
        mltree_root = self._init_mltree(run_settings.initial_state, run_settings.initial_node)
        mcmc_chain = [run_settings.initial_state]

        try:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                self._job_handler = jobhandling.JobHandler(executor, self._models, num_threads)
                self._logger.log_header(self._run_statistics)
                self._update_run_statistics(mcmc_chain)
                self._logger.log_run_statistics(self._run_statistics)
                self._logger.log_debug_new_samples(len(mcmc_chain))

                # --- Main MCMC Loop ---
                while True:
                    self._extend_tree_and_launch_jobs(mltree_root)
                    self._update_tree_from_finished_jobs(mltree_root)
                    self._compute_available_mcmc_decisions(mltree_root)
                    self._mltree_modifier.compress_resolved_subchains(mltree_root)
                    mcmc_chain, mltree_root = self._propagate_chain(mcmc_chain, mltree_root)

                    if len(mcmc_chain) >= self._num_samples:
                        break

        except BaseException as exc:
            self._logger.exception(exc)
            try:
                self._export_debug_tree(mltree_root)
            except RecursionError as exc:
                self._logger.exception(exc)
        finally:
            return mcmc_chain, mltree_root

    # ----------------------------------------------------------------------------------------------
    def get_rngs(self) -> RNGStates:
        rng_states = RNGStates(
            proposal=self._ground_proposal.rng,
            mltree=self._mltree_modifier.rng,
            node_init=self._rng_node_init,
        )
        return rng_states

    # ----------------------------------------------------------------------------------------------
    def set_rngs(self, rng_states: RNGStates) -> None:
        self._ground_proposal.rng = rng_states.proposal
        self._mltree_modifier.rng = rng_states.mltree
        self._rng_node_init = rng_states.node_init

    # ----------------------------------------------------------------------------------------------
    def _init_mltree(self, initial_state: np.ndarray, initial_node) -> mltree.MTNode:
        mltree_root = mltree.MTNode(name="a")
        mltree_root.level = self._num_levels - 1
        mltree_root.subchain_index = 0

        if initial_node is not None:
            mltree_root.state = initial_node.state
            mltree_root.logposterior = initial_node.logposterior
            mltree_root.random_draw = initial_node.random_draw
        else:
            mltree_root.state = initial_state
            mltree_root.random_draw = self._rng_node_init.uniform(low=0, high=1, size=None)
        

        return mltree_root

    # ----------------------------------------------------------------------------------------------
    def _extend_tree_and_launch_jobs(self, mltree_root: mltree.MTNode) -> None:
        while (
            mltree_root.height <= self._maximum_tree_height
        ) and self._job_handler.workers_available:
            self._mltree_modifier.expand_tree(mltree_root)
            self._mltree_modifier.update_probability_reached(
                mltree_root, self._accept_rate_estimator
            )
            new_candidate = mltree_search.find_max_probability_node(mltree_root)
            self._job_handler.submit_job(new_candidate)
            self._log_debug_statistics("submitted", new_candidate)
            self._export_debug_tree(mltree_root)

    # ----------------------------------------------------------------------------------------------
    def _update_tree_from_finished_jobs(self, mltree_root) -> None:
        results, nodes = self._job_handler.get_finished_jobs()
        for result, node in zip(results, nodes):
            if result < self._underflow_threshold:
                node.parent = None
                self._log_debug_statistics("discarded", node)
            else:
                node.logposterior = result
                self._mltree_modifier.update_descendants(node)
                self._log_debug_statistics("returned", node)
            self._export_debug_tree(mltree_root)

    # ----------------------------------------------------------------------------------------------
    def _compute_available_mcmc_decisions(self, mltree_root: mltree.MTNode) -> None:
        computing_mcmc_decisions = True

        while computing_mcmc_decisions:
            computing_mcmc_decisions = False

            for level_children in atree.LevelOrderGroupIter(mltree_root):
                for node in level_children:
                    (
                        node_available_for_decision,
                        is_ground_level_decision,
                        is_two_level_decision,
                    ) = mltree_search.check_if_node_is_available_for_decision(node)

                    if node_available_for_decision:
                        if is_ground_level_decision:
                            accepted = self._mcmc_kernel.compute_single_level_decision(node)
                            self._log_debug_statistics(f"1lmcmc: {accepted}", node)
                        elif is_two_level_decision:
                            same_level_parent = mltree_search.get_same_level_parent(node)
                            accepted = self._mcmc_kernel.compute_two_level_decision(
                                node, same_level_parent
                            )
                            self._log_debug_statistics(f"2lmcmc: {accepted}", node)
                        self._num_generated_samples[node.level] += 1
                        if accepted:
                            self._num_accepted_samples[node.level] += 1
                        self._accept_rate_estimator.update(accepted, node)
                        self._mltree_modifier.discard_rejected_nodes(node, accepted)
                        computing_mcmc_decisions = True

                    if computing_mcmc_decisions:
                        self._export_debug_tree(mltree_root)
                        break
                if computing_mcmc_decisions:
                    break

    # ----------------------------------------------------------------------------------------------
    def _propagate_chain(
        self, mcmc_chain: Sequence[np.ndarray], mltree_root: mltree.MTNode
    ) -> tuple[Sequence[np.ndarray], mltree.MTNode]:
        while (
            unique_child := mltree_search.get_unique_same_subchain_child(mltree_root)
        ) is not None:
            mcmc_chain.append(unique_child.state)
            self._log_run_statistics(mcmc_chain)
            unique_child.parent = None
            mltree_root = unique_child
            self._logger.log_debug_new_samples(len(mcmc_chain))
            self._export_debug_tree(mltree_root)

        return mcmc_chain, mltree_root

    # ----------------------------------------------------------------------------------------------
    def _init_statistics(self) -> tuple[dict[str, logging.Statistic], dict[str, logging.Statistic]]:
        run_statistics = {}
        run_statistics["time"] = logging.Statistic(f"{'Time[s]':<12}", "<12.3e")
        run_statistics["num_samples"] = logging.Statistic(f"{'#Samples':<12}", "<12.3e")
        for i in range(self._num_levels):
            run_statistics[f"evals_l{i}"] = logging.Statistic(f"{f'#Evals L{i}':<12}", "<12.3e")
        for i in range(self._num_levels):
            run_statistics[f"accept_rate_l{i}"] = logging.Statistic(f"{f'AR L{i}':<12}", "<12.3e")
        for i in range(self._num_levels):
            run_statistics[f"ar_estimate_l{i}"] = logging.Statistic(f"{f'ARE L{i}':<12}", "<12.3e")
        
        debug_statistics = {}
        debug_statistics["level"] = logging.Statistic(f"{'level':<6}", "<3")
        debug_statistics["index"] = logging.Statistic(f"{'index':<6}", "<3")
        debug_statistics["state"] = logging.Statistic(f"{'state':<6}", "<12.3e")
        debug_statistics["draw"] = logging.Statistic(f"{'draw':<5}", "<5.3f")
        debug_statistics["logp"] = logging.Statistic(f"{'logp':<5}", "<12.3e")
        debug_statistics["reached"] = logging.Statistic(f"{'reached':<8}", "<12.3e")

        return run_statistics, debug_statistics

    # ----------------------------------------------------------------------------------------------
    def _update_run_statistics(self, mcmc_chain: list[np.ndarray]) -> dict[str, logging.Statistic]:
        self._run_statistics["time"].set_value(time.time() - self._start_time)
        self._run_statistics["num_samples"].set_value(len(mcmc_chain))
        for i in range(self._num_levels):
            self._run_statistics[f"evals_l{i}"].set_value(self._job_handler.num_evaluations[i])
        for i in range(self._num_levels):
            if self._num_generated_samples[i] == 0:
                accept_rate = 0
            else:
                accept_rate = self._num_accepted_samples[i] / self._num_generated_samples[i]
            self._run_statistics[f"accept_rate_l{i}"].set_value(accept_rate)
        for i in range(self._num_levels):
            self._run_statistics[f"ar_estimate_l{i}"].set_value(
                self._accept_rate_estimator.get_acceptance_rate(i)
            )

        return self._run_statistics

    # ----------------------------------------------------------------------------------------------
    def _update_debug_statistics(self, node: mltree.MTNode) -> dict[str, logging.Statistic]:
        self._debug_statistics["level"].set_value(node.level)
        self._debug_statistics["index"].set_value(node.subchain_index)
        self._debug_statistics["state"].set_value(node.state)
        self._debug_statistics["draw"].set_value(node.random_draw)
        self._debug_statistics["logp"].set_value(node.logposterior)
        self._debug_statistics["reached"].set_value(node.probability_reached)

        return self._debug_statistics

    # ----------------------------------------------------------------------------------------------
    def _log_run_statistics(self, mcmc_chain: Sequence[np.ndarray]) -> None:
        if (len(mcmc_chain) % self._print_interval == 0) or (len(mcmc_chain) == self._num_samples):
            self._run_statistics = self._update_run_statistics(mcmc_chain)
            self._logger.log_run_statistics(self._run_statistics)

    # ----------------------------------------------------------------------------------------------
    def _log_debug_statistics(self, info: str, node: mltree.MTNode) -> None:
        self._debug_statistics = self._update_debug_statistics(node)
        self._logger.log_debug_statistics(info, self._debug_statistics)

    # ----------------------------------------------------------------------------------------------
    def _export_debug_tree(self, root: mltree.MTNode) -> None:
        tree_id = self._mltree_visualizer.export_to_dot(root)
        self._logger.log_debug_tree_export(tree_id)
