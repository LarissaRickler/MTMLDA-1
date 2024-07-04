"""_summary_."""

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
    """Settings for initialization of the MLDASampler.

    Attributes:
        num_levels (int): Number of levels in the MLDA hierarchy.
        subsampling_rates (list): List of subchain lengths for each level.
        max_tree_height (int): Maximum height of the tree, purely technical, default is 50.
        underflow_threshold (float): Threshold for underflow of log-posterior values, below
            posterior probability is treated as zero, default is -1000.
        rng_seed_mltree (int): Seed for the random number generator used in the MLTreeModifier.
        rng_seed_node_init (int): Seed for the random number generator used for initialization
            of the the first node in the Markov tree.
        mltree_path (str): Path to the directory where the Markov trees are to be exported as dot
            files. Default is none, meaning that no trees are exported.
    """

    num_levels: int
    subsampling_rates: list
    max_tree_height: int = 50
    underflow_threshold: float = -1000
    rng_seed_mltree: int = 0
    rng_seed_node_init: int = 0
    mltree_path: str = None


@dataclass
class SamplerRunSettings:
    """Settings for conduction of a run with an initialized MLDASampler.

    Attributes:
        num_samples (int): Number of samples to be generated on the fine level.
        initial_state (np.ndarray): Initial state of the Markov chain.
        initial_node (mltree.MTNode): Initial node of the Markov tree, if not given, the initial
            state is used to create the first node. Can be used for reinitialization.
            Default is None.
        num_threads (int): Number of threads to be used for parallel evaluation of posterior
            evaluation requests. Default is 1.
        print_interval (int): Interval for printing run statistics, default is 1.
        tree_render_interval (int): Interval for exporting the Markov tree as a dot file,
            default is 1. Only relevant if an `mltree_path` has been provided during initialization.
    """

    num_samples: int
    initial_state: np.ndarray
    initial_node: mltree.MTNode = None
    num_threads: int = 1
    print_interval: int = 1
    tree_render_interval: int = 1


@dataclass
class RNGStates:
    """Collection of random number generators of different sub-components.

    Can be returned for reproducibility of runs.
    """

    proposal: np.random.Generator
    mltree: np.random.Generator
    node_init: np.random.Generator


# ==================================================================================================
class MTMLDASampler:
    """Main object for parallel MLDA sampling.

    the MTMLDASampler is the main component of the MLDA code base. It is a composite object, taking
    arguments for the logger, a model hierarchy, and accept rate estimator, and a proposal.
    In addition, it internally initilizes a Metropolis-Hastings Kernel object and everything related
    to working with Markov trees for prefetching. The `run` method produces a Markov chain for the
    fine level posterior by repeatedly looping through a number of steps:
    1. Extend the Markov tree by adding new nodes.
    2. Launch jobs for evaluating the posterior at the new nodes.
    3. Update the tree with the results of the finished jobs.
    4. Compute available MCMC decisions.
    5. Propagate the chain by accepting or rejecting nodes.

    Evaluation calls are implemented via a job handler, which is a wrapper around a
    ThreadPoolExecutor object.

    Methods:
        run: Main method for running the sampler.
        get_rngs: Get the random number generators used in the sampler.
        set_rngs: Set the random number generators used in the sampler.
    """

    def __init__(
        self,
        setup_settings: SamplerSetupSettings,
        logger_settings: logging.LoggerSettings,
        models: Sequence[Callable],
        accept_rate_estimator: mcmc.BaseAcceptRateEstimator,
        ground_proposal: mcmc.BaseProposal,
    ) -> None:
        """Constructor of the MLDA sampler.

        The constructor asssembles the compisite sampler object. Components for which multiple
        options might be necessary are passed as arguments, while components that are unique are
        created internally.

        Args:
            setup_settings (SamplerSetupSettings): Data class with settings for sampler setup
            logger_settings (logging.LoggerSettings): Data class with settings for the logger
            models (Sequence[Callable]): List of callables resembling the MLDA model hierarchy
            accept_rate_estimator (mcmc.BaseAcceptRateEstimator): Accept rate estimator
                for prefetching
            ground_proposal (mcmc.BaseProposal): Proposal object for thr ground level MCMC chain
        """
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

        self._num_generated_samples = [
            0,
        ] * self._num_levels
        self._num_accepted_samples = [
            0,
        ] * self._num_levels

    # ----------------------------------------------------------------------------------------------
    def run(self, run_settings: SamplerRunSettings) -> list[np.ndarray]:
        """Main  run methods for the sampler.

        The method takes a data class with run-specific settings. As described in the construtor,
        it initiates a while loop that continuous until the desired number of fine level samples
        has been generated. This method is mainly an interface, most of the program logic is
        implemented in the private sub-methods.

        Args:
            run_settings (SamplerRunSettings): Settings for the sampler run.

        Returns:
            list[np.ndarray,mltree.MTNode]: Produced Markov chain and the final Markov tree root.
        """
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
        """Get the random number generators used in the algorithm."""
        rng_states = RNGStates(
            proposal=self._ground_proposal.rng,
            mltree=self._mltree_modifier.rng,
            node_init=self._rng_node_init,
        )
        return rng_states

    # ----------------------------------------------------------------------------------------------
    def set_rngs(self, rng_states: RNGStates) -> None:
        """Set the random number generators used in the algorithm."""
        self._ground_proposal.rng = rng_states.proposal
        self._mltree_modifier.rng = rng_states.mltree
        self._rng_node_init = rng_states.node_init

    # ----------------------------------------------------------------------------------------------
    def _init_mltree(
        self,
        initial_state: Union[np.ndarray, None] = None,
        initial_node: Union[mltree.MTNode, None] = None,
    ) -> mltree.MTNode:
        """Initialize the Markov tree for prefetching.

        The method either takes an initial state or a Markov tree node. In the first case, a new
        tree is initialized from the state alone. If a node is provided, its logposterior and
        random draw is also taken over to the new tree.

        Args:
            initial_state (np.ndarray): Initial state of the Markov chain.
            initial_node (mltree.MTNode): Markov tree node to be utilized as initial state.

        Returns:
            mltree.MTNode: Root of the Markov tree.
        """
        assert (initial_state is not None) or (
            initial_node is not None
        ), "Either initial state or initial node must be provided."

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
        """Expands the Markov tree and requests evaluation of posterior for new nodes.

        The methods uses the `MLTreeModifier` to expand the Markov tree and requests new evaluations
        from the `JobHandler`. Jobs requested correspond to those most likely needed in accordance
        with the acceptance rate estimates.

        Args:
            mltree_root (mltree.MTNode): Root of the Markov tree to extend
        """
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
        """Get finished jobs from job handler, update Markov tree accordingly.

        If a job returns with log-probablity lower than the underflow threshold, the corresponding
        node and its descendants are discarded, since they correspond to states that are impossible
        to reach. Otherwise, the log-probability is transferred to the node and its descendants are
        updated accordingly. 

        Args:
            mltree_root (mltree.MTNode): Current Markov tree, only necessary for visualization.
        """
        results, nodes = self._job_handler.get_finished_jobs()
        for result, node in zip(results, nodes):
            if result < self._underflow_threshold:
                # Zero probability -> Discard this tree branch
                node.parent = None
                self._log_debug_statistics("discarded", node)
            else:
                # Transfer information from computation
                node.logposterior = result
                self._mltree_modifier.update_descendants(node)
                self._log_debug_statistics("returned", node)
            self._export_debug_tree(mltree_root)

    # ----------------------------------------------------------------------------------------------
    def _compute_available_mcmc_decisions(self, mltree_root: mltree.MTNode) -> None:
        """Traverse the Markov tree repeatedly and compute available MCMC decisions.

        This includes one-level and two-level descisions. As the tree structure is modified after
        an MCMC decision, it is completely traversed after each decision. The acceptance rates for
        each level are updated accordingly.

        Args:
            mltree_root (mltree.MTNode): Markov tree root to compute decisions from.
        """
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
                            # One-level decision for ground level
                            accepted = self._mcmc_kernel.compute_single_level_decision(node)
                            self._log_debug_statistics(f"1lmcmc: {accepted}", node)
                        elif is_two_level_decision:
                            # Two-level decision upwards
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
        """Propagate chain from one fine level sample to the next.

        Propagation takes place if a unique path is available between the corresponding nodes in the
        Markov tree. The new sample becomes the root of the Markov tree for the next iteration. The
        procedure is repeated until no more unique paths can be found.

        Args:
            mcmc_chain (Sequence[np.ndarray]): List of fine lavel samples
            mltree_root (mltree.MTNode): current Markov tree root

        Returns:
            tuple[Sequence[np.ndarray], mltree.MTNode]: Updated Markov chain and tree root
        """
        while (unique_child := mltree_search.get_unique_same_level_child(mltree_root)) is not None:
            mcmc_chain.append(unique_child.state)
            self._log_run_statistics(mcmc_chain)
            unique_child.parent = None
            mltree_root = unique_child
            self._logger.log_debug_new_samples(len(mcmc_chain))
            self._export_debug_tree(mltree_root)

        return mcmc_chain, mltree_root

    # ----------------------------------------------------------------------------------------------
    def _init_statistics(self) -> tuple[dict[str, logging.Statistic], dict[str, logging.Statistic]]:
        """Initialize statistics dictionaries.

        The MTMLDA logger takes to types of statistis: Run statistics are logged in a table, debug
        statistics contain more detailed information, logged to a separate file. All statistics are
        stored in the respective dictionaries with a string handle, a string id for the output, and
        the format the corresponding value should be printed in.

        Returns:
            tuple[dict[str, logging.Statistic], dict[str, logging.Statistic]]:
                Debug and run statistics
        """
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
        """Update values in the run statistics dictionary.

        Args:
            mcmc_chain (list[np.ndarray]): Current MCMC chain

        Returns:
            dict[str, logging.Statistic]: Updateted statistics dictionary
        """
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
        """Update values in the debug statistics dictionary.

        Args:
            node (mltree.MTNode): Markov tree node to extract debug information from

        Returns:
            dict[str, logging.Statistic]: Updateted statistics dictionary
        """
        self._debug_statistics["level"].set_value(node.level)
        self._debug_statistics["index"].set_value(node.subchain_index)
        self._debug_statistics["state"].set_value(node.state)
        self._debug_statistics["draw"].set_value(node.random_draw)
        self._debug_statistics["logp"].set_value(node.logposterior)
        self._debug_statistics["reached"].set_value(node.probability_reached)

        return self._debug_statistics

    # ----------------------------------------------------------------------------------------------
    def _log_run_statistics(self, mcmc_chain: Sequence[np.ndarray]) -> None:
        """Print out run statistics to table.

        Only executes if the number of samples is a multiple of the print interval or the last

        Args:
            mcmc_chain (Sequence[np.ndarray]): current MCMC chain
        """
        if (len(mcmc_chain) % self._print_interval == 0) or (len(mcmc_chain) == self._num_samples):
            self._run_statistics = self._update_run_statistics(mcmc_chain)
            self._logger.log_run_statistics(self._run_statistics)

    # ----------------------------------------------------------------------------------------------
    def _log_debug_statistics(self, info: str, node: mltree.MTNode) -> None:
        """Print out debug statistics to separate file.

        Args:
            info (str): String name for event to be logged
            node (mltree.MTNode): Node to log debug info for
        """
        self._debug_statistics = self._update_debug_statistics(node)
        self._logger.log_debug_statistics(info, self._debug_statistics)

    # ----------------------------------------------------------------------------------------------
    def _export_debug_tree(self, root: mltree.MTNode) -> None:
        """Export Markov tree dot file and log event.

        Args:
            root (mltree.MTNode): Root of the Markov tree to export
        """
        tree_id = self._mltree_visualizer.export_to_dot(root)
        self._logger.log_debug_tree_export(tree_id)
