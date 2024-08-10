import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import umbridge as ub

import src.mtmlda.mcmc as mcmc
import src.mtmlda.utilities as utils
from components import abstract_builder, posterior, prior


# ==================================================================================================
@dataclass
class InverseProblemSettings(abstract_builder.InverseProblemSettings):
    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    prior_rng_seed: int
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
    pass


# ==================================================================================================
class ApplicationBuilder(abstract_builder.ApplicationBuilder):
    # ----------------------------------------------------------------------------------------------
    def __init__(self, process_id: int) -> None:
        super().__init__(process_id)
        self._prior_component = None

    # ----------------------------------------------------------------------------------------------
    def set_up_models(self, inverse_problem_settings: InverseProblemSettings) -> list[Callable]:
        server_available = False
        while not server_available:
            try:
                if self._process_id == 0:
                    print("Calling server...")
                likelihood_component = ub.HTTPModel(
                    inverse_problem_settings.ub_model_address,
                    inverse_problem_settings.ub_model_name,
                )
                if self._process_id == 0:
                    print("Server available\n")
                server_available = True
            except Exception as exc:
                print(exc)
                time.sleep(10)

        prior_rng_seed = utils.distribute_rng_seeds_to_processes(
            inverse_problem_settings.prior_rng_seed, self._process_id
        )
        prior_component = prior.GaussianLogPrior(
            inverse_problem_settings.prior_mean,
            inverse_problem_settings.prior_covariance,
            prior_rng_seed,
        )
        self._prior_component = prior_component

        model_wrapper = posterior.LogPosterior(prior_component, likelihood_component)
        models = [
            partial(model_wrapper, config=config)
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
        initial_state = self._prior_component.sample()
        return initial_state
