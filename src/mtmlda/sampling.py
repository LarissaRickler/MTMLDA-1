import logging
import os
import sys
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anytree as atree
import numpy as np

from . import jobhandling, mcmc, mltree
from .mltree import MLTreeSearchFunctions as mltree_search


# ==================================================================================================
@dataclass(kw_only=True)
class SamplerSetupSettings:
    num_levels: int
    subsampling_rates: list
    max_tree_height: int
    underflow_threshold: float
    rng_seed_mltree: int
    rng_seed_node_init: int
    do_printing: bool
    mltree_path: str
    logfile_path: str
    write_mode: str


@dataclass(kw_only=True)
class SamplerRunSettings:
    num_samples: int
    initial_state: np.ndarray
    num_threads: int
    print_interval: int
    tree_render_interval: int


@dataclass(kw_only=True)
class RNGStates:
    proposal: np.random.Generator
    mltree: np.random.Generator
    node_init: np.random.Generator


# ==================================================================================================
class MTMLDASampler:
    def __init__(
        self,
        setup_settings: SamplerSetupSettings,
        models: Sequence[Callable],
        accept_rate_estimator: mcmc.BaseAcceptRateEstimator,
        ground_proposal: mcmc.BaseProposal,
    ) -> None:
        self._num_levels = setup_settings.num_levels
        self._models = models
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
        self._logger = self._init_logging(
            setup_settings.do_printing, setup_settings.logfile_path, setup_settings.write_mode
        )

        self._start_time = None
        self._num_samples = None
        self._print_interval = None
        self._tree_render_interval = None
        self._job_handler = None
        self._logging_components = None

    # ----------------------------------------------------------------------------------------------
    def run(self, run_settings: SamplerRunSettings) -> list[np.ndarray]:
        self._start_time = time.time()
        self._num_samples = run_settings.num_samples
        self._print_interval = run_settings.print_interval
        self._tree_render_interval = run_settings.tree_render_interval
        num_threads = run_settings.num_threads

        mltree_root = self._init_mltree(run_settings.initial_state)
        mcmc_chain = [run_settings.initial_state]

        try:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                self._job_handler = jobhandling.JobHandler(executor, self._models, num_threads)
                self._logger.print_header()
                logging_components = self._get_logging_components(mcmc_chain)
                self._logger.print_statistics(logging_components)

                # --- Main MCMC Loop ---
                while True:
                    self._extend_tree_and_launch_jobs(mltree_root)
                    self._update_tree_from_finished_jobs()
                    self._compute_available_mcmc_decisions(mltree_root)
                    self._mltree_modifier.compress_resolved_subchains(mltree_root)
                    mcmc_chain, mltree_root = self._propagate_chain(mcmc_chain, mltree_root)

                    if len(mcmc_chain) >= self._num_samples:
                        break

        except BaseException as exc:
            self._logger.exception(exc)
            try:
                self._mltree_visualizer.export_to_dot(mltree_root, id="exc")
            except RecursionError as exc:
                self._logger.exception(exc)
        finally:
            return mcmc_chain

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
    def _init_mltree(self, initial_state: np.ndarray) -> mltree.MTNode:
        mltree_root = mltree.MTNode(name="a")
        mltree_root.state = initial_state
        mltree_root.random_draw = self._rng_node_init.uniform(low=0, high=1, size=None)
        mltree_root.level = self._num_levels - 1
        mltree_root.subchain_index = 0

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

    # ----------------------------------------------------------------------------------------------
    def _update_tree_from_finished_jobs(self) -> None:
        results, nodes = self._job_handler.get_finished_jobs()
        for result, node in zip(results, nodes):
            if result < self._underflow_threshold:
                node.parent = None
            else:
                node.logposterior = result
                self._mltree_modifier.update_descendants(node)

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
                        elif is_two_level_decision:
                            same_level_parent = mltree_search.get_same_level_parent(node)
                            accepted = self._mcmc_kernel.compute_two_level_decision(
                                node, same_level_parent
                            )
                        self._accept_rate_estimator.update(accepted, node)
                        self._mltree_modifier.discard_rejected_nodes(node, accepted)
                        computing_mcmc_decisions = True

                    if computing_mcmc_decisions:
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
            mcmc_chain.append(mltree_root.state)
            self._generate_output(mcmc_chain, mltree_root)
            unique_child.parent = None
            mltree_root = unique_child

        return mcmc_chain, mltree_root

    # ----------------------------------------------------------------------------------------------
    def _init_logging(self, do_printing, logfile_path, write_mode) -> None:
        logging_components = {}
        logging_components["time"] = {"id": "Time[s]", "width": 12, "format": "12.3e"}
        logging_components["samples"] = {"id": "#Samples", "width": 12, "format": "12.3e"}
        for i in range(self._num_levels):
            logging_components[f"evals_l{i}"] = {
                "id": f"#Evals L{i}",
                "width": 12,
                "format": "12.3e",
            }
        logger = MTMLDALogger(do_printing, logfile_path, write_mode, logging_components)

        return logger

    # ----------------------------------------------------------------------------------------------
    def _get_logging_components(self, mcmc_chain: list[np.ndarray]) -> dict[str, Any]:
        logging_components = {}
        logging_components["time"] = time.time() - self._start_time
        logging_components["samples"] = len(mcmc_chain)

        for i in range(self._num_levels):
            logging_components[f"evals_l{i}"] = self._job_handler.num_evaluations[i]

        return logging_components

    # ----------------------------------------------------------------------------------------------
    def _generate_output(
        self, mcmc_chain: Sequence[np.ndarray], mltree_root: mltree.MTNode
    ) -> None:
        if (len(mcmc_chain) % self._print_interval == 0) or (len(mcmc_chain) == self._num_samples):
            logging_components = self._get_logging_components(mcmc_chain)
            self._logger.print_statistics(logging_components)

        if (len(mcmc_chain) % self._tree_render_interval == 0) or (
            len(mcmc_chain) == self._num_samples
        ):
            self._mltree_visualizer.export_to_dot(mltree_root, id=f"{len(mcmc_chain)}")


# ==================================================================================================
class MTMLDALogger:

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        do_printing: bool,
        logfile_path: Path,
        write_mode: str,
        components: dict[str, dict[str, Any]] = None,
    ) -> None:
        self._components = components
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
    def print_header(self) -> None:
        header_str = ""
        for component in self._components.keys():
            component_name = self._components[component]["id"]
            component_width = self._components[component]["width"]
            header_str += f"{component_name:{component_width}}| "
        separator = "-" * len(header_str)
        self._pylogger.info(header_str)
        self._pylogger.info(separator)

    # ----------------------------------------------------------------------------------------------
    def print_statistics(self, info_dict: dict[str, Any]) -> None:
        output_str = ""

        for component, value in info_dict.items():
            component_format = self._components[component]["format"]
            output_str += f"{value:<{component_format}}| "
        self._pylogger.info(output_str)

    # ----------------------------------------------------------------------------------------------
    def info(self, message: str) -> None:
        self._pylogger.info(message)

    # ----------------------------------------------------------------------------------------------
    def exception(self, message: str) -> None:
        self._pylogger.exception(message)
