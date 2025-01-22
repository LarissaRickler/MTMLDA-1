"""Base class for application builders.

The MTMLDA takes a list of callables as posterior hierarchy. To enhance flexibility w.r.t. to how
these models are generated, and to support parallel chains, we employ a builder pattern for
applications. With the prescribed interface, the builder can be used by the run wrapper and trigger
MTMLDA run from a variety of use-cases.

Classes:
    ApplicationBuilder: Base class for application builders.
    InverseProblemSettings: Base data class for inverse problem settings.
    SamplerComponentSettings: Base data class for sampler component settings.
    InitialStateSettings: Base data class for initial state settings.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mtmlda.core import mcmc


# ==================================================================================================
@dataclass
class InverseProblemSettings:
    """Empty base data class for inverse problem settings."""


@dataclass
class SamplerComponentSettings:
    """Empty base data class for sampler component settings."""


@dataclass
class InitialStateSettings:
    """Empty base data class for initial state settings."""


# ==================================================================================================
class ApplicationBuilder(ABC):
    """Base class for application builders.

    The class provides a generic interface for application builders, as will be invoked by the run
    wrapper. The class is purely abstract, implementation must follow in subclasses. The only
    concrete behavior the class enforces is that the id of the invoking process is provided. This
    is useful for setting up components in a process-specific way for chain-parallel runs.

    Methods:
        set_up_models: Set up a model hierarchy in the form of a list of callables..
        set_up_sampler_components: Set up proposal and accept rate estimator.
        generate_initial_state: Generate the initial state for the Markov chain
    """

    def __init__(self, process_id: int) -> None:
        """Base class constructor, assigns process id of calling process as internal attribute.

        Args:
            process_id (int): ID of calling process
        """
        self._process_id = process_id

    @abstractmethod
    def set_up_models(self, inverse_problem_settings: InverseProblemSettings) -> list[Callable]:
        """Set up a posterior hierarchy as a sequence of callables.

        Within this method, different components can be freely combined to obtain a posterior
        hierarchy. For instance, an external model server could be called to provide a likelihood,
        which is then combined with a prior component. This method is the main reason the separate
        builder exists, to allow for more flexibility in the model setup.

        Args:
            inverse_problem_settings (InverseProblemSettings): Necessary settings for model setup,
                can be anything

        Returns:
            list[Callable]: Posterior hierarchy for MLDA run
        """
        raise NotImplementedError

    @abstractmethod
    def set_up_sampler_components(
        self, sampler_component_settings: SamplerComponentSettings
    ) -> tuple[mcmc.BaseProposal, mcmc.BaseAcceptRateEstimator]:
        """Set up the coarse-level proposal and accept rate estimator for MLDA.

        Can be used for process-dependent initialization.

        Args:
            sampler_component_settings (SamplerComponentSettings): Necessary settings for component
                setup, can be anything

        Returns:
            tuple[mcmc.BaseProposal, mcmc.BaseAcceptRateEstimator]: Proposal and
                accept rate estimator
        """
        raise NotImplementedError

    @abstractmethod
    def generate_initial_state(self, initial_state_settings: InitialStateSettings) -> np.ndarray:
        """Generate initial state of a Markov chain.

        Can be used for process-dependent initialization.

        Args:
            initial_state_settings (InitialStateSettings): Necessary settings for initial state
                setup, can be anything

        Returns:
            np.ndarray: Initial state
        """
        raise NotImplementedError
