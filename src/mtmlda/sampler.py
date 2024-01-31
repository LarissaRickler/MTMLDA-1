from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import os
import sys
import time

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
        self._subsampling_rates = setup_settings.subsampling_rates
        self._maximum_tree_height = setup_settings.max_tree_height
        self._accept_rate_estimator = accept_rate_estimator
        self._mcmc_kernel = MLMetropolisHastingsKernel(ground_proposal)
        self._mltree_modifier = MLTreeModifier(
            setup_settings.num_levels,
            ground_proposal,
            setup_settings.subsampling_rates,
            setup_settings.rng_seed,
        )
        self._mltree_visualizer = MLTreeVisualizer(setup_settings.mltree_path)
        self._logger = MTMLDALogger(
            setup_settings.do_printing, setup_settings.logfile_path, setup_settings.write_mode
        )
        self._job_handler = None
        self._start_time = None
        self._num_samples = None
        self._print_interval = None
        self._tree_render_interval = None

    # ----------------------------------------------------------------------------------------------
    def run(self, run_settings):
        self._start_time = time.time()
        num_threads = run_settings.num_threads
        self._num_samples = run_settings.num_samples
        self._print_interval = run_settings.print_interval
        self._tree_render_interval = run_settings.tree_render_interval

        mltree_root = self._init_mltree(run_settings.initial_state, run_settings.rng_seed)
        mcmc_chain = [mltree_root.state]
        self._logger.print_statistics(
            print_header=True, samples=len(mcmc_chain), time=time.time()-self._start_time
        )

        try:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                self._job_handler = JobHandler(executor, self._models, num_threads)

                # --- Main MCMC Loop ---
                while True:
                    self._extend_tree_and_launch_jobs(mltree_root)
                    self._update_tree_from_finished_jobs()
                    self._mltree_modifier.propagate_log_posterior_to_reject_children(mltree_root)
                    self._compute_available_mcmc_decisions(mltree_root)
                    self._mltree_modifier.compress_resolved_subchains(mltree_root)

                    while (
                        unique_child := search_functions.get_unique_fine_level_child(
                            mltree_root, self._num_levels
                        )
                    ) is not None:
                        self._generate_output(mcmc_chain, mltree_root)
                        mcmc_chain.append(mltree_root.state)
                        unique_child.parent = None
                        mltree_root = unique_child

                    if len(mcmc_chain) >= self._num_samples:
                        break

        except BaseException as exc:
            self._logger.exception(exc)
            try:
                self._mltree_visualizer.export_to_dot(mltree_root)
            except RecursionError as exc:
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
    def _extend_tree_and_launch_jobs(self, mltree_root):
        while (
            mltree_root.height <= self._maximum_tree_height
        ) and self._job_handler.workers_available:
            self._mltree_modifier.expand_tree(mltree_root)
            self._mltree_modifier.update_probability_reached(
                mltree_root, self._accept_rate_estimator
            )
            new_candidate = search_functions.find_max_probability_node(mltree_root)
            self._job_handler.submit_job(new_candidate)

    # ----------------------------------------------------------------------------------------------
    def _update_tree_from_finished_jobs(self):
        results, nodes = self._job_handler.get_finished_jobs()
        for result, node in zip(results, nodes):
            node.logposterior = result

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

    # ----------------------------------------------------------------------------------------------
    def _generate_output(self, mcmc_chain, mltree_root):  #
        if (len(mcmc_chain) % self._print_interval == 0) or (len(mcmc_chain) == self._num_samples):
            self._logger.print_statistics(
                samples=len(mcmc_chain), time=time.time()-self._start_time
            )

        if (len(mcmc_chain) % self._tree_render_interval == 0) or (
            len(mcmc_chain) == self._num_samples
        ):
            self._mltree_visualizer.export_to_dot(mltree_root)
        return mcmc_chain


# ==================================================================================================
class MTMLDALogger:
    _print_formats = {
        "samples": {"id": "#Samples", "width": 12, "format": "12.3e"},
        "time": {"id": "Time[s]", "width": 12, "format": "12.3e"},
    }

    # ----------------------------------------------------------------------------------------------
    def __init__(self, do_printing, logfile_path, write_mode):
        self._pylogger = logging.getLogger(__name__)
        self._pylogger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(message)s")

        if not self._pylogger.hasHandlers():
            if do_printing:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter
                console_handler.setFormatter(formatter)
                self._pylogger.addHandler(console_handler)

            if logfile_path is not None:
                os.makedirs(logfile_path.parent, exist_ok=True)
                file_handler = logging.FileHandler(logfile_path, mode=write_mode)
                file_handler.setFormatter(formatter)
                self._pylogger.addHandler(file_handler)

    # ----------------------------------------------------------------------------------------------
    def print_statistics(self, print_header=False, **kwargs):
        if print_header:
            self._print_header(**kwargs)
        output_str = ""

        for component, statistic in kwargs.items():
            component_format = self._print_formats[component]["format"]
            output_str += f"{statistic:<{component_format}}| "
        self._pylogger.info(output_str)

    # ----------------------------------------------------------------------------------------------
    def info(self, message):
        self._pylogger.info(message)

    # ----------------------------------------------------------------------------------------------
    def exception(self, message):
        self._pylogger.exception(message)

    # ----------------------------------------------------------------------------------------------
    def _print_header(self, **kwargs):
        header_str = ""
        for component in kwargs.keys():
            component_name = self._print_formats[component]["id"]
            component_width = self._print_formats[component]["width"]
            header_str += f"{component_name:{component_width}}| "
        header_width = len(header_str)
        separator = "-" * header_width
        self._pylogger.info(header_str)
        self._pylogger.info(separator)


# ==================================================================================================
@dataclass
class SamplerSetupSettings:
    num_levels: int
    subsampling_rates: list
    max_tree_height: int
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
    tree_render_interval: int
