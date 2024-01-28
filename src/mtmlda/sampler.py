from concurrent.futures import ThreadPoolExecutor

import numpy as np
from anytree import LevelOrderGroupIter, util

from mlmcmc import MLMetropolisHastingsKernel
from .mltree import MTNode
from .jobhandling import JobHandler


class MTMLDASampler:
    def __init__(self, mtmlda_settings, mtmlda_components):
        self._num_levels = mtmlda_settings.num_levels
        self._subsampling_rates = mtmlda_settings.sub_sampling_rates
        self._models = mtmlda_components.models
        self._proposals = mtmlda_components.proposals
        self._accept_rate_estimator = mtmlda_components.accept_rate_estimator
        self._mltree_searcher = mtmlda_components.mltree_searcher
        self._mltree_modifier = mtmlda_components.mltree_modifier
        self._mlmcmc_kernel = MLMetropolisHastingsKernel()
        self._job_handler = None

    def run(self, run_settings):
        num_samples = run_settings.num_samples
        initial_state = run_settings.initial_state
        num_workers = run_settings.num_workers
        rng = np.random.default_rng(run_settings.rng_seed)

        mltree_root = MTNode("a")
        mltree_root.state = initial_state
        mltree_root.random_draw = rng.random.uniform()
        mltree_root.level = len(self._num_levels) - 1
        mltree_root.subchain_index = 0

        mcmc_chain = [mltree_root.state]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            self._job_handler = JobHandler(executor)
            for _ in range(num_workers):
                self._choose_and_submit_job(mltree_root)

            # --- Main MCMC Loop ---
            while True:
                self._update_tree_from_finished_jobs(mltree_root)
                self._compute_available_mcmc_decisions(mltree_root)
                
                unique_child = self._mltree_searcherget_unique_fine_level_child(mltree_root)
                if unique_child is not None:
                    mcmc_chain.append(mltree_root.state)
                    unique_child.parent = None
                    mltree_root = unique_child
                if len(mcmc_chain) >= num_samples:
                    break
                while self._job_handler.num_busy_workers < num_workers:
                    self._choose_and_submit_job(mltree_root)


    def _choose_and_submit_job(self, mltree_root):
        self._mltree_modifier.expand_tree(mltree_root)
        self._mltree_modifier.update_probability_reached(mltree_root, self._accept_rate_estimator)
        most_promising_candidate = self._mltree_searcher.find_max_probability_node(mltree_root)
        self._job_handler.submit_job(most_promising_candidate)

    def _update_tree_from_finished_jobs(self, mltree_root):
        results, nodes = self._job_handler.get_finished_jobs()
        for result, node in zip(results, nodes):
            node.logposterior = result
            self._mltree_modifier.propagate_log_posterior_to_reject_children(node)

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
                        accepted = self._mlmcmc_kernel.compute_single_level_decision(node)
                        self._accept_rate_estimator.update(accepted, node.level)
                        self._discard_rejected_nodes(node, accepted)
                        trying_to_compute_mcmc_decision = True

                    if node_available_for_decision and is_two_level_decision:
                        same_level_parent = self._mltree_searcher.get_same_level_parent(node)
                        accepted = self._mlmcmc_kernel.compute_two_level_decision(
                            node, same_level_parent
                        )
                        self._discard_rejected_nodes(node, accepted)
                        trying_to_compute_mcmc_decision = True

                if trying_to_compute_mcmc_decision:
                    break
            if trying_to_compute_mcmc_decision:
                break

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
            same_level_parent = self._mltree_searcher.get_same_level_parent(node)
            same_level_parent_child_on_path = (
                self._mltree_searcher.get_same_level_parent_child_on_path(node)
            )
            is_ground_level_decision = node.level == node.parent.level == 0
            is_two_level_decision = (
                node.level - 1 == node.parent.level
                and same_level_parent.logposterior is not None
                and same_level_parent_child_on_path.logposterior is not None
            )
        return node_available_for_decision, is_ground_level_decision, is_two_level_decision

    def _discard_rejected_nodes(self, node, accepted):
        if accepted:
            util.rightsibling(node).parent = None
        else:
            node.parent = None
