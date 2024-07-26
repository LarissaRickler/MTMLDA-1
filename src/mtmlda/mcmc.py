"""MLDA-specific MCMC routines.

This module contains all MCMC-specific functionalities, including those necessary in the Multilevel
context. Roughly speaking, this comprises proposals for coarse level changes and accept-reject
routines. The only prefetching-specific component is the accept rate estimator, which is used to
predict possible future states of the Markov chain.
Porposals and accept rate estimators are implemenkted in class hierarchies. New options can easily
be implemented by inheriting from the respective base class interfaces.

Classes:
    BaseProposal: Base class for MCMC proposals
    RandomWalkProposal: Metropolis random walk proposal
    PCNProposal: Preconditioned Crank-Nicolson proposal
    BaseAcceptRateEstimator: Base class for MLDA accept rate estimators
    StaticAcceptRateEstimator: Static accept rate estimator
    MLMetropolisHastingsKernel: Metropolis-Hastings acceptance kernel for multilevel decisions
"""

from abc import abstractmethod
from numbers import Real
from typing import Any

import numpy as np

from . import mltree


# ==================================================================================================
class BaseProposal:
    """Base class for MCMC proposals.

    This class defines the interface for MCMC proposals. It is an abstract class and cannot be
    instantiated. It provides two abstract methods that need to be implemented by subclasses:
    propose and evaluate_log_probability. The propose method generates a new proposal based on the
    current state. The evaluate_log_probability method evaluates the log probability of a new move
    depending on the current state.

    Methods:
        propose: Propose new MCMC move
        evaluate_log_probability: Evaluate log probability of a new move depending on the 
            current state
    Attributes:
        rng: Random number generator used for proposals
    """

    def __init__(self, seed: int) -> None:
        """Base class constructor.

        The constructor simply initializes a numpy random number generator using the provided seed.

        Args:
            seed (int): Seed value for the random number generator

        Raises:
            TypeError: Checks for valid seed type
        """
        if not isinstance(seed, int):
            raise TypeError("Proposal seed must be an integer")
        self._rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def propose(self, current_state: np.ndarray) -> None:
        """Abstract method for proposal generation.

        This method needs to be re-implemented in child classes.

        Args:
            current_state (np.ndarray): State of the Markov chain to propose from
        """
        pass

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_log_probability(self, left_state: np.ndarray, right_state: np.ndarray) -> None:
        """Abstract method for log probability evaluation.

        This method needs to be re-implemented in child classes. The states are not called "current"
        and "new" state, as their relation is not as clear in the multilevel setting. The names
        simply refer to the order of arguments in the evaluation formula.

        Args:
            left_state (np.ndarray): First state for evaluation
            right_state (np.ndarray): Second state for evaluation
        """
        pass

    # ----------------------------------------------------------------------------------------------
    @property
    def rng(self) -> np.random.Generator:
        """Getter of the random number generator for proposals.

        Returns:
            np.random.Generator: Numpy RNG object
        """
        return self._rng

    # ----------------------------------------------------------------------------------------------
    @rng.setter
    def rng(self, rng: np.random.Generator) -> None:
        """Setter of the random number generator for proposals.

        Args:
            rng (np.random.Generator): Numpy RNG object
        """
        self._rng = rng


# ==================================================================================================
class RandomWalkProposal(BaseProposal):
    """Random walk proposal.
    
    Implementation of the Metropolis random walk proposal, inheriting from the BaseProposal class.
    The proposal distribution is a Gaussian about the current state. We utilize a fixed covariance
    matrix and step width. Note that the proposal distribution is symmetric w.r.t. to its arguments,
    so that it vanishes in the Metropolis-Hastings acceptance ratio. Therefore, the
    `evaluate_log_probability` method only returns 0 as a dummy value.
    """

    def __init__(self, step_width: float, covariance: np.ndarray, seed: int) -> None:
        """Proposal constructor.

        The constructor takes a fixed step width and covariance matrix for the Gaussian proposal.

        Args:
            step_width (float): Proposal step width , needs to be positive
            covariance (np.ndarray): Proposal covariance matrix
            seed (int): Seed for initialization of the proposal RNG

        Raises:
            ValueError: Checks if step width is a positive real number
            ValueError: Checks if covariance matrix can be Cholesky factorized
        """
        super().__init__(seed)
        if not isinstance(step_width, Real) and step_width > 0:
            raise ValueError("Step width must be a positive real number")
        self._step_width = step_width
        try:
            self._cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as err:
            raise ValueError("Cannot Cholesky factorize covariance matrix") from err

    # ----------------------------------------------------------------------------------------------
    def propose(self, current_state: np.ndarray) -> np.ndarray:
        r"""Propose new move from current state.

        Given the covariance :math:`\\Sigma` and  current state :math:`m`, the new state is
        proposed from the distribution :math:`\\mathcal{N}(m, \\Sigma)`.


        Args:
            current_state (np.ndarray): Current state of the Markov chain to propose from

        Returns:
            np.ndarray: Proposal state
        """
        assert current_state.size == self._cholesky.shape[0], "State and covariance size mismatch"
        standard_normal_increment = self._rng.normal(size=current_state.size)
        proposal = current_state + self._step_width * self._cholesky @ standard_normal_increment
        return proposal

    # ----------------------------------------------------------------------------------------------
    def evaluate_log_probability(self, proposal: np.ndarray, current_state: np.ndarray) -> float:
        """Evaluate log probability of proposal, given current state.

        This is only a dummy returning zero, as the conditional proposal probabilities vanish
        in the Metropolis-Hastings acceptance ratio.

        Args:
            proposal (np.ndarray): Proposal state
            current_state (np.ndarray): Current state

        Returns:
            float: Dummy log-probability, here always zero
        """
        return 0


# ==================================================================================================
class PCNProposal(BaseProposal):
    """Preconditioned Crank-Nicolson random walk proposal.
    
    Implementation of the pCN proposal, inheriting from the BaseProposal class.
    The pCN proposal is an asymmetric extension of the Metropolis random walk proposal. It has been
    developed particularly for high-dimensional parameter spaces.
    """

    def __init__(self, beta: float, covariance: np.ndarray, seed: int) -> None:
        """Proposal constructor.

        The parameter `beta` is analogous to the step with in the Metropolis random walk proposal.
        It determines the "explicitness" of the underlying CN scheme.

        Args:
            beta (float): Step width, in (0,1)
            covariance (np.ndarray): Covariance matrix for Gaussian
            seed (int): RNG seed

        Raises:
            ValueError: Checks that `beta` is in (0,1)
            ValueError: Checks that covariance matrix can be Cholesky factorized
            ValueError: Checks that covariance matrix can be inverted
        """
        super().__init__(seed)
        if not isinstance(beta, Real) and beta > 0 and beta < 1:
            raise ValueError("beta must be in (0,1)")
        self._beta = beta
        try:
            self._cholesky = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as err:
            raise ValueError("Cannot Cholesky factorize covariance matrix") from err
        try:
            self._precision = np.linalg.inv(covariance)
        except np.linalg.LinAlgError as err:
            raise ValueError("Cannot invert covariance matrix") from err

    # ----------------------------------------------------------------------------------------------
    def propose(self, current_state: np.ndarray) -> np.ndarray:
        r"""Propose a new state from the current one.

        Given the covariance :math:`\\Sigma` and  current state :math:`m`, the new state is
        proposed from the distribution :math:`\\mathcal{N}(\\sqrt{1-\\beta^2}m, \\beta^2\\Sigma)`.

        Args:
            current_state (np.ndarray): Current state of the Markov chain

        Returns:
            np.ndarray: New proposal
        """
        assert current_state.size == self._cholesky.shape[0], "State and covariance size mismatch"
        standard_normal_increment = self._rng.normal(size=current_state.size)
        proposal = (
            np.sqrt(1 - self._beta**2) * current_state
            + self._beta * self._cholesky @ standard_normal_increment
        )
        return proposal

    # ----------------------------------------------------------------------------------------------
    def evaluate_log_probability(self, proposal: np.ndarray, current_state: np.ndarray) -> float:
        r"""Evaluate the log probability of the proposal given the current state.

        The log probability for a proposal :math:`m'` from current state :math:`m` is given by
        .. math::
            \Delta m = m' - \sqrt{1-\beta^2}m,
            \log p = -\frac{1}{2} \Delta m^T \beta^2 \Sigma^{-1} \Delta m.

        Args:
            proposal (np.ndarray): Proposal state
            current_state (np.ndarray): Current state

        Returns:
            float: Log probability of the proposal given the current state
        """
        assert proposal.size == self._cholesky.shape[0], "Proposal and covariance size mismatch"
        assert current_state.size == self._cholesky.shape[0], "State and covariance size mismatch"
        state_diff = proposal - np.sqrt(1 - self._beta**2) * current_state
        log_probability = -0.5 * state_diff.T @ (self._beta**2 * self._precision) @ state_diff
        return log_probability


# ==================================================================================================
class BaseAcceptRateEstimator:
    """Base class for MLDA accept rate estimators.
    
    This class defines the interface for accept rate estimators. It is an abstract class and cannot
    be instantiated. The prefetching approach to our MLDA implementation requires a-priori estimates
    for accepting MCMC moves on the different levels of the model hierarchy. Such estimates can 
    depend on any number of factors like the level itself, the current state, or the sample history.
    In the simplest case, the acceptance rates are constant.

    Methods:
        get_acceptance_rate: Get the acceptance rate estimate for a given MCMC move
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def get_acceptance_rate(self, *args: Any, **kwargs: Any) -> float:
        """Abstract getter for the acceptance rate."""
        pass


# ==================================================================================================
class StaticAcceptRateEstimator(BaseAcceptRateEstimator):
    """Static accept rate estimator.
    
    This class implements a simple accept rate estimator based on a fixed update scheme. The object
    is initialized with a list of estimates for each levels. The estimates are then incremented or
    decremented after each MCMC decision, depending on the acceptance of the move.

    Methods:
        get_acceptance_rate: Get the acceptance rate estimate for a given MCMC move
        update: Update the acceptance rate estimate after a MCMC move
    """

    def __init__(self, initial_guess: list[float], update_parameter: float) -> None:
        """Estimator Constructor.

        The constructor takes a list of initial guesses for each level and a fixed parameter for the
        update rule.

        Args:
            initial_guess (list[float]): List of initial guesses for the acceptance rates of each
                level
            update_parameter (float): Update parameter 

        Raises:
            ValueError: Checks that initial guesses are in (0,1)
            ValueError: CHecks that update parameter is in (0,1)
        """
        for init_guess in initial_guess:
            if not isinstance(init_guess, Real) and init_guess > 0 and init_guess < 1:
                raise ValueError("Initial guess must be in (0,1)")
        if not isinstance(update_parameter, Real) and update_parameter > 0 and update_parameter < 1:
            raise ValueError("Update parameter must be in (0,1)")
        self._acceptance_rates = initial_guess
        self._update_parameter = update_parameter

    # ----------------------------------------------------------------------------------------------
    def get_acceptance_rate(self, level: int) -> float:
        """Return acceptance rate.

        For this simple estimator, the acceptance rate only depends on the current level.

        Args:
            level (int): Level in the MLDA algorithm

        Returns:
            float: Acceptance rate estimate
        """
        assert level < len(self._acceptance_rates), "Level out of range"
        acceptance_rate = self._acceptance_rates[level]
        return acceptance_rate

    # ----------------------------------------------------------------------------------------------
    def update(self, accepted: bool, node: mltree.MTNode) -> None:
        r"""Update an acceptance rate estimate.

        Given a node, the update modifies the estimate corresponding to the node's level.
        Let :math:`\\alpha` be the current estimate and :math:`\\delta` the update parameter.
        If the corresponding MCMC move has been accepted, the new estimate is
        :math:`(1-\\delta)\\alpha + \\delta`. Otherwise, the new estimate is
        :math:`(1-\\delta)\\alpha`.

        Args:
            accepted (bool): Bool telling if the MCMC move has been accepted
            node (mltree.MTNode): Node from which the move has been performed
        """
        level = node.level
        assert level < len(self._acceptance_rates), "Level out of range"
        decreased_rate = (1 - self._update_parameter) * self._acceptance_rates[level]
        if accepted:
            self._acceptance_rates[level] = decreased_rate + self._update_parameter
        else:
            self._acceptance_rates[level] = decreased_rate


# ==================================================================================================
class MLMetropolisHastingsKernel:
    """Metropolis-Hastings Acceptance Kernel.
    
    The kernel implements the Metroplis-Hastings acceptance rule for the multilevel setting.
    Accordingly, to different acceptance rules are implemented. One for within-level moves, utilized
    for standard MCMC on the coarsest level of the model hierarchy. The other is for between-level
    moves on the higher levels.

    Methods:
        compute_single_level_decision: Compute the decision for a within-level MCMC move
        compute_two_level_decision: Compute the decision for a between-level MCMC move
    """

    def __init__(self, ground_proposal: BaseProposal) -> None:
        """Kernel constructor.

        Takes an object derived from BaseProposal.

        Args:
            ground_proposal (BaseProposal): Proposal object
        """
        self._ground_proposal = ground_proposal

    # ----------------------------------------------------------------------------------------------
    def compute_single_level_decision(self, node: mltree.MTNode) -> bool:
        """Compute within level MCMC decision.

        This is the standard Metropolis-Hastings acceptance rule for single-level MCMC.

        Args:
            node (mltree.MTNode): Node whose state contains the proposal

        Returns:
            bool: If proposal has been accepted
        """
        new_state = node.state
        old_state = node.parent.state
        posterior_logp_new = node.logposterior
        posterior_logp_old = node.parent.logposterior
        proposal_logp_new_old = self._ground_proposal.evaluate_log_probability(new_state, old_state)
        proposal_logp_old_new = self._ground_proposal.evaluate_log_probability(old_state, new_state)

        assert all(
            value is not None for value in (posterior_logp_new, posterior_logp_old)
        ), "Posterior value is None"

        accept_probability = min(
            1,
            np.exp(
                posterior_logp_new
                + proposal_logp_old_new
                - posterior_logp_old
                - proposal_logp_new_old
            ),
        )
        accepted = node.parent.random_draw < accept_probability
        return accepted

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def compute_two_level_decision(node: mltree.MTNode, same_level_parent: mltree.MTNode) -> bool:
        """Compute between level MCMC decision.

        This decision rule is specific to the multilevel setting. Note that it does not utilize the
        proposal, but only the node containing the proposal and its same-level parent.

        Args:
            node (mltree.MTNode): Node containing the proposal
            same_level_parent (mltree.MTNode): Same level parent of that node

        Returns:
            bool: If proposal has been accepted
        """
        posterior_logp_new_fine = node.logposterior
        posterior_logp_old_coarse = same_level_parent.children[0].logposterior
        posterior_logp_old_fine = same_level_parent.logposterior
        posterior_logp_new_coarse = node.parent.logposterior

        assert all(
            value is not None
            for value in (
                posterior_logp_new_fine,
                posterior_logp_old_coarse,
                posterior_logp_old_fine,
                posterior_logp_new_coarse,
            )
        ), "Posterior value is None"

        accept_probability = min(
            1,
            np.exp(
                posterior_logp_new_fine
                + posterior_logp_old_coarse
                - posterior_logp_old_fine
                - posterior_logp_new_coarse
            ),
        )
        accepted = node.parent.random_draw < accept_probability
        return accepted
