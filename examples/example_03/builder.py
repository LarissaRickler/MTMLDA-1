from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from mtmlda import utilities as utils
from mtmlda.components import abstract_builder
from mtmlda.core import mcmc


# ==================================================================================================
@dataclass
class InverseProblemSettings(abstract_builder.InverseProblemSettings):
    ub_model_configs: dict[str, str]
    ub_model_address: str
    ub_model_name: str


@dataclass
class SamplerComponentSettings(abstract_builder.SamplerComponentSettings):
    proposal_step_width: float
    proposal_covariance: np.ndarray
    proposal_rng_seed: int
    accept_rates_initial_guess: list[float]
    accept_rates_update_parameter: float


@dataclass
class InitialStateSettings(abstract_builder.InitialStateSettings):
    mean_init: np.ndarray
    covariance_init: np.ndarray
    rng_seed_init: int


# ==================================================================================================
class ApplicationBuilder(abstract_builder.ApplicationBuilder):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, process_id: int) -> None:
        super().__init__(process_id)
        self._prior_component = None

    # ----------------------------------------------------------------------------------------------
    def set_up_models(self, inverse_problem_settings: InverseProblemSettings) -> list[Callable]:
        posterior_component = utils.request_umbridge_server(
            self._process_id,
            inverse_problem_settings.ub_model_address,
            inverse_problem_settings.ub_model_name,
        )

        models = [
            partial(posterior_component, config=config)
            for config in inverse_problem_settings.ub_model_configs
        ]

        return models

    # ----------------------------------------------------------------------------------------------
    def set_up_sampler_components(
        self, sampler_component_settings: SamplerComponentSettings
    ) -> tuple[Any, Any]:
        proposal_rng_seed = utils.distribute_rng_seeds_to_processes(
            sampler_component_settings.proposal_rng_seed, self._process_id
        )
        ground_proposal = mcmc.RandomWalkProposal(
            sampler_component_settings.proposal_step_width,
            sampler_component_settings.proposal_covariance,
            proposal_rng_seed,
        )

        accept_rate_estimator = mcmc.StaticAcceptRateEstimator(
            sampler_component_settings.accept_rates_initial_guess,
            sampler_component_settings.accept_rates_update_parameter,
        )

        return ground_proposal, accept_rate_estimator

    # ----------------------------------------------------------------------------------------------
    def generate_initial_state(self, initial_state_settings: InitialStateSettings) -> np.ndarray:
        rng_seed = utils.distribute_rng_seeds_to_processes(
            initial_state_settings.rng_seed_init, self._process_id
        )
        init_rng = np.random.default_rng(rng_seed)
        initial_state = init_rng.multivariate_normal(
            initial_state_settings.mean_init, initial_state_settings.covariance_init
        )

        return initial_state
