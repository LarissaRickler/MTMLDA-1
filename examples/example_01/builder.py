import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import umbridge as ub

import src.mtmlda.mcmc as mcmc
from components import abstract_builder


# ==================================================================================================
@dataclass
class InverseProblemSettings(abstract_builder.InverseProblemSettings):
    ub_model_address: str
    ub_model_names: str


@dataclass
class SamplerComponentSettings(abstract_builder.SamplerComponentSettings):
    proposal_step_width: float
    proposal_covariance: np.ndarray
    proposal_rng_seed: int
    accept_rates_initial_guess: list[float]
    accept_rates_update_parameter: float


@dataclass
class InitialStateSettings(abstract_builder.InitialStateSettings):
    initial_states: list[np.ndarray]


# ==================================================================================================
class ApplicationBuilder(abstract_builder.ApplicationBuilder):

    # ----------------------------------------------------------------------------------------------
    def __init__(self, process_id: int) -> None:
        super().__init__(process_id)

    # ----------------------------------------------------------------------------------------------
    def set_up_models(self, inverse_problem_settings: InverseProblemSettings) -> list[Callable]:
        server_available = False
        while not server_available:
            try:
                if self._process_id == 0:
                    print("Calling server...")
                posterior_models = [
                    ub.HTTPModel(
                        inverse_problem_settings.ub_model_address,
                        inverse_problem_settings.ub_model_names[0],
                    ),
                    ub.HTTPModel(
                        inverse_problem_settings.ub_model_address,
                        inverse_problem_settings.ub_model_names[1],
                    ),
                    ub.HTTPModel(
                        inverse_problem_settings.ub_model_address,
                        inverse_problem_settings.ub_model_names[2],
                    ),
                ]
                if self._process_id == 0:
                    print("Server available\n")
                server_available = True
            except Exception as exc:
                print(exc)
                time.sleep(10)

        return posterior_models

    # ----------------------------------------------------------------------------------------------
    def set_up_sampler_components(
        self, sampler_component_settings: SamplerComponentSettings
    ) -> tuple[Any, Any]:
        sampler_component_settings.proposal_rng_seed = self._process_id
        ground_proposal = mcmc.RandomWalkProposal(
            sampler_component_settings.proposal_step_width,
            sampler_component_settings.proposal_covariance,
            sampler_component_settings.proposal_rng_seed,
        )

        accept_rate_estimator = mcmc.MLAcceptRateEstimator(
            sampler_component_settings.accept_rates_initial_guess,
            sampler_component_settings.accept_rates_update_parameter,
        )

        return ground_proposal, accept_rate_estimator

    # ----------------------------------------------------------------------------------------------
    def generate_initial_state(self, initial_state_settings: InitialStateSettings) -> np.ndarray:
        initial_state = initial_state_settings.initial_states[self._process_id]
        return initial_state
