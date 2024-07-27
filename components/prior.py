"""Prior components.

These prior implementations adhere to a common interface provided by the abstract base class.
New prior distributions can be implemented through subclassing.

Classes:
    BaseLogPrior: Base class for prior distributions.
    UniformLogPrior: Implementation of a uniform prior.
    GaussianLogPrior: Implementation of Gaussian prior.
"""

from abc import ABC, abstractmethod

import numpy as np


# ==================================================================================================
class BaseLogPrior(ABC):
    """Base class for prior distributions.

    This class prescribes the basic interface for a prior distribution, as required by other
    components. Basically, a prior needs to provide methods to evaluate its log-likelihood, and
    to draw samples from it. The base class also provides an UM-Bridge like call interface for the
    evaluation of the log-probability.

    Methods:
        __call__: UM-Bridge-like call interface for the log-prior
        evaluate: Evaluate the log-probability for a given parameter vector
        sample: Draw a sample from the prior
    """

    @abstractmethod
    def __init__(self, seed: int) -> None:
        """Base class constructor, takin seed for the internal random number generator.

        Args:
            seed (int): RNG seed
        """
        self._rng = np.random.default_rng(seed)

    def __call__(self, parameter: list[list[float]]) -> list[list[float]]:
        """UM-Bridge-like call interface for log-probability evaluation.

        This method simply converts the input parameter to a numpy array and delegates the call to
        the `evaluate` method. the output is again transformed to the UM-Bridge  format.

        Args:
            parameter (list[list[float]]): Parameter candidate

        Returns:
            list[list[float]]: Log-probability value
        """
        parameter = np.array(parameter[0])
        return [[self.evaluate(parameter)]]

    @abstractmethod
    def evaluate(self, parameter: np.ndarray) -> float:
        """Compute log-probability for given parameter."""
        pass

    @abstractmethod
    def sample(self) -> np.ndarray:
        """Draw a sample from the prior."""
        pass


# ==================================================================================================
class UniformLogPrior(BaseLogPrior):
    """Implementation of a uniform prior."""

    def __init__(self, parameter_intervals: np.ndarray, seed: int) -> None:
        """Constructor.

        Args:
            parameter_intervals (np.ndarray): Bounds for each parameter
            seed (int): RNG seed
        """
        super().__init__(seed)
        self._lower_bounds = parameter_intervals[:, 0]
        self._upper_bounds = parameter_intervals[:, 1]
        self._interval_lengths = self._upper_bounds - self._lower_bounds

    def evaluate(self, parameter: np.ndarray) -> float:
        """Compute log-probability for given parameter.

        Note that the prior simply returns 0 if the parameter is within the bounds, and -inf
        otherwise. This is because a uniform prior enters into the posterior only as a constant,
        which is irrelevant in MCMC.
        """
        has_support = ((parameter >= self._lower_bounds) & (parameter <= self._upper_bounds)).all()

        if has_support:
            return 0
        else:
            return -np.inf

    def sample(self) -> np.ndarray:
        """Draw a sample from the prior."""
        sample = self._rng.uniform(self._lower_bounds, self._upper_bounds)
        return sample


# ==================================================================================================
class GaussianLogPrior(BaseLogPrior):
    """Implementation of Gaussian prior."""

    def __init__(self, mean: np.ndarray, covariance: np.ndarray, seed: int) -> None:
        """Constructor.

        Args:
            mean (np.ndarray): Mean vector
            covariance (np.ndarray): Covariance matrix
            seed (int): RNG seed
        """
        super().__init__(seed)
        self._mean = mean
        self._cholesky = np.linalg.cholesky(covariance)
        self._precision = np.linalg.inv(covariance)

    def evaluate(self, parameter: np.array) -> float:
        """Compute log-probability for given parameter."""
        parameter_diff = parameter - self._mean
        log_probability = -0.5 * parameter_diff.T @ self._precision @ parameter_diff
        return log_probability

    def sample(self) -> np.ndarray:
        """Draw a sample from the prior."""
        standard_normal_increment = self._rng.normal(size=self._mean.size)
        sample = self._mean + self._cholesky @ standard_normal_increment
        return sample
