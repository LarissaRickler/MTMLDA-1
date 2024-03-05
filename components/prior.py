from abc import abstractmethod

import numpy as np


# ==================================================================================================
class BaseLogPrior:
    @abstractmethod
    def __init__(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def __call__(self, parameter: list[list[float]]) -> list[list[float]]:
        parameter = np.array(parameter[0])
        return [[self.evaluate(parameter)]]

    @abstractmethod
    def evaluate(self, parameter: np.ndarray) -> float:
        pass


# ==================================================================================================
class UniformLogPrior(BaseLogPrior):
    def __init__(self, parameter_intervals: np.ndarray, seed: int) -> None:
        super().__init__(seed)
        self._lower_bounds = parameter_intervals[:, 0]
        self._upper_bounds = parameter_intervals[:, 1]
        self._interval_lengths = self._upper_bounds - self._lower_bounds

    def evaluate(self, parameter: np.ndarray) -> float:
        has_support = ((parameter >= self._lower_bounds) & (parameter <= self._upper_bounds)).all()

        if has_support:
            return 0
        else:
            return -np.inf
        

# ==================================================================================================
class GaussianLogPrior(BaseLogPrior):
    def __init__(self, mean: np.ndarray, covariance: np.ndarray, seed) -> None:
        super().__init__(seed)
        self._mean = mean
        self._precision = np.linalg.inv(covariance)

    def evaluate(self, parameter: np.array) -> float:
        parameter_diff = parameter - self._mean
        log_probability = -0.5 * parameter_diff.T @ self._precision @ parameter_diff
        return log_probability
        