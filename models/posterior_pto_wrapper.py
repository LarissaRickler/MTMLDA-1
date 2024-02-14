from typing import Any

import numpy as np
import umbridge as ub


class UninformLogPrior:
    is_const = True

    def __init__(self, parameter_intervals: np.ndarray, seed: int = 0) -> None:
        self._lower_bounds = parameter_intervals[:, 0]
        self._upper_bounds = parameter_intervals[:, 1]
        self._interval_lengths = self._upper_bounds - self._lower_bounds
        self._rng = np.random.default_rng(seed)
        self._log_prior_const = -1

    def evaluate(self, parameter: np.ndarray) -> float:
        parameter = np.array(parameter[0])
        has_support = ((parameter >= self._lower_bounds) & (parameter <= self._upper_bounds)).all()

        if has_support:
            return self._log_prior_const
        else:
            return -np.inf
        
    def sample(self):
        sample = self._rng.uniform(self._lower_bounds, self._upper_bounds)
        return sample


class GaussianLogLikelihood:
    def __init__(
        self, umbridge_pto_map: ub.Model, data: np.ndarray, covariance: np.ndarray
    ) -> None:
        self._umbridge_pto_map = umbridge_pto_map
        self._data = data
        self._precision = np.linalg.inv(covariance)

    def evaluate(self, parameter: list[list[float]], config) -> float:
        observables = np.array(self._umbridge_pto_map(parameter, config)[0])
        misfit = self._data - observables
        log_likelihood = -0.5 * misfit.T @ self._precision @ misfit
        return log_likelihood


class LogPosterior:
    def __init__(self, log_prior: Any, log_likelihood: Any) -> None:
        self._log_prior = log_prior
        self._log_likelihood = log_likelihood

    def __call__(self, parameter: np.ndarray, config) -> list[list[float]]:
        return [[self.evaluate(parameter, config)]]

    def evaluate(self, parameter: np.ndarray, config) -> float:
        log_prior = self._log_prior.evaluate(parameter)
        if np.isneginf(log_prior):
            log_posterior = -np.inf
        else:
            log_posterior = self._log_likelihood.evaluate(parameter, config)
            if not self._log_prior.is_const:
                log_posterior += log_prior

        return log_posterior
