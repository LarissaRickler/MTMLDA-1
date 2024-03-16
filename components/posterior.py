from typing import Any

import numpy as np
import umbridge as ub


# ==================================================================================================
class GaussianLLFromPTOMap:
    def __init__(
        self, umbridge_pto_map: ub.Model, data: np.ndarray, covariance: np.ndarray
    ) -> None:
        self._umbridge_pto_map = umbridge_pto_map
        self._data = data
        self._precision = np.linalg.inv(covariance)

    def __call__(self, parameter: list[list[float]], config: dict[str, Any]) -> list[list[float]]:
        observables = np.array(self._umbridge_pto_map(parameter, config)[0])
        misfit = self._data - observables
        log_likelihood = -0.5 * misfit.T @ self._precision @ misfit
        return [[log_likelihood]]


# ==================================================================================================
class LogPosterior:
    def __init__(self, log_prior: Any, log_likelihood: Any) -> None:
        self._log_prior = log_prior
        self._log_likelihood = log_likelihood

    def __call__(self, parameter: list[list[float]], **log_likelihood_args) -> list[list[float]]:
        log_prior = self._log_prior(parameter)
        if np.isneginf(log_prior[0][0]):
            log_posterior = log_prior
        else:
            log_likelihood = self._log_likelihood(parameter, **log_likelihood_args)
            log_posterior = [[log_likelihood[0][0] + log_prior[0][0]]]

        return log_posterior
        