from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
import sys

import numpy as np
from anytree import LevelOrderGroupIter, util

from .mlmcmc import MLMetropolisHastingsKernel
from .mltree import (
    MTNode,
    MLTreeModifier,
    MLTreeVisualizer,
    MLTreeSearchFunctions as search_functions,
)
from .jobhandling import JobHandler


# ==================================================================================================
class MTMLDASampler:
    def __init__(self, setup_settings, models, accept_rate_estimator, ground_proposal):
        self._num_levels = setup_settings.num_levels
        self._models = models
        self._accept_rate_estimator = accept_rate_estimator
        self._mcmc_kernel = MLMetropolisHastingsKernel(ground_proposal)
        self._mltree_modifier = MLTreeModifier(
            ground_proposal, setup_settings.subsampling_rates, setup_settings.rng_seed
        )
        self._mltree_visualizer = MLTreeVisualizer(setup_settings.mltree_path)
        self._logger = MTMLDALogger(
            setup_settings.do_printing, setup_settings.logfile_path, setup_settings.write_mode
        )
        self._job_handler = None

    # ----------------------------------------------------------------------------------------------
    def run(self, run_settings):
        num_samples = run_settings.num_samples
        num_threads = run_settings.num_threads
        print_interval = run_settings.print_interval

        self._logger.print_header()
        mltree_root = self._init_mltree(run_settings.initial_state, run_settings.rng_seed)
        mcmc_chain = [mltree_root.state]
        self._logger.print_statistics({"samples": len(mcmc_chain)})

        try:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                self._job_handler = JobHandler(executor, self._models)
                for _ in range(num_threads):
                    self._choose_and_submit_job(mltree_root)

                # --- Main MCMC Loop ---
                while True:
                    self._update_tree_from_finished_jobs()
                    self._compute_available_mcmc_decisions(mltree_root)

                    while (
                        unique_child := search_functions.get_unique_fine_level_child(
                            mltree_root, self._num_levels
                        )
                        is not None
                    ):
                        mcmc_chain.append(mltree_root.state)
                        self._mltree_visualizer.export_to_dot(mltree_root)
                        unique_child.parent = None
                        mltree_root = unique_child
                        if (len(mcmc_chain) % print_interval == 0) or (
                            len(mcmc_chain) == num_samples
                        ):
                            self._logger.print_statistics({"samples": len(mcmc_chain)})

                    if len(mcmc_chain) >= num_samples:
                        break
                    while self._job_handler.num_busy_workers < num_threads:
                        self._choose_and_submit_job(mltree_root)
        except BaseException as exc:
            self._logger.exception(exc)
        finally:
            return mcmc_chain

    # ----------------------------------------------------------------------------------------------
    def _init_mltree(self, initial_state, seed):
        rng = np.random.default_rng(seed)
        mltree_root = MTNode("a")
        mltree_root.state = initial_state
        mltree_root.random_draw = rng.uniform()
        mltree_root.level = self._num_levels - 1
        mltree_root.subchain_index = 0

        return mltree_root

    # ----------------------------------------------------------------------------------------------
    def _choose_and_submit_job(self, mltree_root):
        self._mltree_modifier.expand_tree(mltree_root)
        self._mltree_modifier.update_probability_reached(mltree_root, self._accept_rate_estimator)
        most_promising_candidate = search_functions.find_max_probability_node(mltree_root)
        if most_promising_candidate is None:
            raise ValueError("No more nodes to compute, most likely tree expansion failed!")
        self._job_handler.submit_job(most_promising_candidate)

    # ----------------------------------------------------------------------------------------------
    def _update_tree_from_finished_jobs(self):
        results, nodes = self._job_handler.get_finished_jobs()
        for result, node in zip(results, nodes):
            node.logposterior = result
            self._mltree_modifier.propagate_log_posterior_to_reject_children(node)

    # ----------------------------------------------------------------------------------------------
    def _compute_available_mcmc_decisions(self, mltree_root):
        trying_to_compute_mcmc_decision = True

        while trying_to_compute_mcmc_decision:
            trying_to_compute_mcmc_decision = False

            for level_children in LevelOrderGroupIter(mltree_root):
                for node in level_children:
                    (
                        node_available_for_decision,
                        is_ground_level_decision,
                        is_two_level_decision,
                    ) = self._check_if_node_is_available_for_decision(node)

                    if node_available_for_decision and is_ground_level_decision:
                        accepted = self._mcmc_kernel.compute_single_level_decision(node)
                        self._accept_rate_estimator.update(accepted, node)
                        self._discard_rejected_nodes(node, accepted)
                        trying_to_compute_mcmc_decision = True

                    if node_available_for_decision and is_two_level_decision:
                        same_level_parent = search_functions.get_same_level_parent(node)
                        accepted = self._mcmc_kernel.compute_two_level_decision(
                            node, same_level_parent
                        )
                        self._accept_rate_estimator.update(accepted, node)
                        self._discard_rejected_nodes(node, accepted)
                        trying_to_compute_mcmc_decision = True

                    if trying_to_compute_mcmc_decision:
                        break
                if trying_to_compute_mcmc_decision:
                    break
    
    # ----------------------------------------------------------------------------------------------
    def _check_if_node_is_available_for_decision(self, node):
        node_available_for_decision = (
            node.name == "a"
            and node.parent is not None
            and len(node.parent.children) > 1
            and node.logposterior is not None
            and node.parent.logposterior is not None
        )

        if not node_available_for_decision:
            is_ground_level_decision = False
            is_two_level_decision = False
        else:
            same_level_parent = search_functions.get_same_level_parent(node)
            is_ground_level_decision = node.level == node.parent.level == 0
            is_two_level_decision = (
                node.level - 1 == node.parent.level
                and same_level_parent.logposterior is not None
                and same_level_parent.children[0].logposterior is not None
            )
        return node_available_for_decision, is_ground_level_decision, is_two_level_decision

    # ----------------------------------------------------------------------------------------------
    def _discard_rejected_nodes(self, node, accepted):
        if accepted:
            util.rightsibling(node).parent = None
        else:
            node.parent = None


# ==================================================================================================
class MTMLDALogger:
    _print_components = ["samples"]

    # ----------------------------------------------------------------------------------------------
    def __init__(self, do_printing, logfile_path, write_mode):
        print_widths = [len(component) for component in self._print_components]
        self._header_string = " ".join(self._print_components)
        self._print_widths = dict(zip(self._print_components, print_widths))
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")

        if not self._logger.hasHandlers():
            if do_printing:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter
                console_handler.setFormatter(formatter)
                self._logger.addHandler(console_handler)

            if logfile_path is not None:
                os.makedirs(logfile_path.parent, exist_ok=True)
                file_handler = logging.FileHandler(logfile_path, mode=write_mode)
                file_handler.setFormatter(formatter)
                self._logger.addHandler(file_handler)

    # ----------------------------------------------------------------------------------------------
    def print_header(self):
        self._logger.info(self._header_string)
        self._logger.info("-" * sum(self._print_widths.values()))

    # ----------------------------------------------------------------------------------------------
    def print_statistics(self, statistics):
        output_str = ""

        for component, statistic in statistics.items():
            output_str += f"{statistic:<{self._print_widths[component]}} "
        self._logger.info(output_str)

    # ----------------------------------------------------------------------------------------------
    def info(self, message):
        self._logger.info(message)

    # ----------------------------------------------------------------------------------------------
    def exception(self, message):
        self._logger.exception(message)


# ==================================================================================================
@dataclass
class SamplerSetupSettings:
    num_levels: int
    subsampling_rates: list
    rng_seed: int
    do_printing: bool
    mltree_path: str
    logfile_path: str
    write_mode: str


@dataclass
class SamplerRunSettings:
    num_samples: int
    initial_state: np.ndarray
    num_threads: int
    rng_seed: int
    print_interval: int
