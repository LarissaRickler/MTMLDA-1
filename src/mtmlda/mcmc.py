from abc import abstractmethod
from typing import Any

import numpy as np

from . import mltree


# ==================================================================================================
class BaseProposal:
    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def propose(self, current_state: np.ndarray) -> None:
        pass

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_log_probability(self, left_state: np.ndarray, right_state: np.ndarray) -> None:
        pass

    # ----------------------------------------------------------------------------------------------
    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    # ----------------------------------------------------------------------------------------------
    @rng.setter
    def rng(self, rng: np.random.Generator) -> None:
        self._rng = rng


# ==================================================================================================
class RandomWalkProposal(BaseProposal):
    def __init__(self, step_width: float, covariance: np.ndarray, seed: int) -> None:
        super().__init__(seed)
        self._step_width = step_width
        self._cholesky = np.linalg.cholesky(covariance)

    # ----------------------------------------------------------------------------------------------
    def propose(self, current_state: np.ndarray) -> np.ndarray:
        standard_normal_increment = self._rng.normal(size=current_state.size)
        proposal = current_state + self._step_width * self._cholesky @ standard_normal_increment
        return proposal

    # ----------------------------------------------------------------------------------------------
    def evaluate_log_probability(self, proposal: np.ndarray, current_state: np.ndarray) -> float:
        return 0


# ==================================================================================================
class PCNProposal(BaseProposal):
    def __init__(self, beta: float, covariance: np.ndarray, seed: int) -> None:
        super().__init__(seed)
        self._beta = beta
        self._cholesky = np.linalg.cholesky(covariance)
        self._precision = np.linalg.inv(covariance)

    # ----------------------------------------------------------------------------------------------
    def propose(self, current_state: np.ndarray) -> np.ndarray:
        standard_normal_increment = self._rng.normal(size=current_state.size)
        proposal = (
            np.sqrt(1 - self._beta**2) * current_state
            + self._beta * self._cholesky @ standard_normal_increment
        )
        return proposal

    # ----------------------------------------------------------------------------------------------
    def evaluate_log_probability(self, proposal: np.ndarray, current_state: np.ndarray) -> float:
        state_diff = proposal - np.sqrt(1 - self._beta**2) * current_state
        log_probability = -0.5 * state_diff.T @ (self._beta**2 * self._precision) @ state_diff
        return log_probability


# ==================================================================================================
class BaseAcceptRateEstimator:
    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def get_acceptance_rate(self, *args: Any, **kwargs: Any) -> float:
        pass


# ==================================================================================================
class StaticAcceptRateEstimator(BaseAcceptRateEstimator):
    def __init__(self, initial_guess: list[float], update_parameter: float) -> None:
        self._acceptance_rates = initial_guess
        self._update_parameter = update_parameter

    # ----------------------------------------------------------------------------------------------
    def get_acceptance_rate(self, level: int) -> float:
        acceptance_rate = self._acceptance_rates[level]
        return acceptance_rate

    # ----------------------------------------------------------------------------------------------
    def update(self, accepted: bool, node: mltree.MTNode) -> None:
        level = node.level
        decreased_rate = (1 - self._update_parameter) * self._acceptance_rates[level]
        if accepted:
            self._acceptance_rates[level] = decreased_rate + self._update_parameter
        else:
            self._acceptance_rates[level] = decreased_rate


# ==================================================================================================
class MLMetropolisHastingsKernel:
    def __init__(self, ground_proposal: BaseProposal) -> None:
        self._ground_proposal = ground_proposal

    # ----------------------------------------------------------------------------------------------
    def compute_single_level_decision(self, node: mltree.MTNode) -> bool:
        new_state = node.state
        old_state = node.parent.state
        posterior_logp_new = node.logposterior
        posterior_logp_old = node.parent.logposterior
        proposal_logp_new_old = self._ground_proposal.evaluate_log_probability(new_state, old_state)
        proposal_logp_old_new = self._ground_proposal.evaluate_log_probability(old_state, new_state)

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
        posterior_logp_new_fine = node.logposterior
        posterior_logp_old_coarse = same_level_parent.children[0].logposterior
        posterior_logp_old_fine = same_level_parent.logposterior
        posterior_logp_new_coarse = node.parent.logposterior

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
