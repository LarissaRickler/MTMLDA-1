import time

from functools import partial

import umbridge as ub

import settings
import src.mtmlda.mcmc as mcmc
from components import builder, prior, posterior


# ==================================================================================================
class ApplicationBuilder(builder.ApplicationBuilder):

    # ----------------------------------------------------------------------------------------------
    def __init__(self, process_id: int):
        super().__init__(process_id)
        self._prior_component = None

    # ----------------------------------------------------------------------------------------------
    def set_up_models(self, inverse_problem_settings: settings.InverseProblemSettings):
        server_available = False
        while not server_available:
            try:
                likelihood_component = ub.HTTPModel(
                    inverse_problem_settings.ub_model_address,
                    inverse_problem_settings.ub_model_name,
                )
                if self._process_id == 0:
                    print("Server available")
                server_available = True
            except:
                if self._process_id == 0:
                    print("Server not available")
                time.sleep(10)

        inverse_problem_settings.rng_seed = self._process_id
        prior_component = prior.GaussianLogPrior(
            inverse_problem_settings.prior_mean,
            inverse_problem_settings.prior_covariance,
            inverse_problem_settings.rng_seed,
        )
        self._prior_component = prior_component

        model_wrapper = posterior.LogPosterior(prior_component, likelihood_component)
        models = [
            partial(model_wrapper, config=mesh_file_name)
            for mesh_file_name in inverse_problem_settings.ub_model_configs
        ]

        return models

    # ----------------------------------------------------------------------------------------------
    def set_up_sampler_components(
        self, sampler_component_settings: settings.SamplerComponentSettings
    ):
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
    def generate_initial_state(self, initial_state_settings: settings.InitialStateSettings):
        initial_state = self._prior_component.sample()
        return initial_state
