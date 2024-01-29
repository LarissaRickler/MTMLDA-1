from abc import abstractmethod
from dataclasses import dataclass

import numpy as np


class BaseProposal:
    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)

    @abstractmethod
    def propose(self, current_state):
        pass

    @abstractmethod
    def evaluate_log_probability(self, left_state, right_state):
        pass

class RandomWalkProposal(BaseProposal):
    def __init__(self, step_width, covariance, seed):
        super().__init__(seed)
        self._step_width = step_width
        self._cholesky = np.linalg.cholesky(covariance)
        self._precision = np.linalg.inv(covariance)

    def propose(self, current_state):
        standard_normal_increment = self._rng.normal(size=current_state.size)
        proposal = current_state + self._step_width * self._cholesky @ standard_normal_increment
        return proposal

    def evaluate_log_probability(self, proposal, current_state):
        state_diff = proposal - current_state
        log_probability = -0.5 * state_diff.T @ self._precision @ state_diff
        return log_probability


class PCNProposal(BaseProposal):
    def __init__(self, beta, covariance, seed):
        super().__init__(seed)
        self._beta = beta
        self._cholesky = np.linalg.cholesky(covariance)
        self._precision = np.linalg.inv(covariance)

    def propose(self, current_state):
        standard_normal_increment = self._rng.normal(size=current_state.size)
        proposal = (
            np.sqrt(1 - self._beta**2) * current_state
            + self._beta * self._cholesky @ standard_normal_increment
        )
        return proposal

    def evaluate_log_probability(self, proposal, current_state):
        state_diff = proposal - np.sqrt(1 - self._beta**2) * current_state
        log_probability = -0.5 * state_diff.T @ (self._beta**2 * self._precision) @ state_diff
        return log_probability


@dataclass
class RandomWalkProposalSettings:
    step_width: float
    covariance: np.ndarray
    rng_seed: int

@dataclass
class PCNProposalSettings:
    beta: float
    covariance: np.ndarray
    rng_seed: int