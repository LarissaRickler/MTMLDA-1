import time
from functools import partial

import umbridge as ub

import src.mtmlda.mcmc as mcmc
from components import prior, posterior


# ==================================================================================================
class ComponentSetup:
    
    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def set_up_models(process_id, model_settings, prior_settings, likelihood_settings):
        configs = model_settings.configs

        server_available = False
        while not server_available:
            try:
                pto_model = ub.HTTPModel(model_settings.address, model_settings.name)
                if process_id == 0:
                    print("Server available")
                server_available = True
            except:
                if process_id == 0:
                    print("Server not available")
                time.sleep(10)

        prior_component = prior.UniformLogPrior(
            prior_settings.parameter_intervals, prior_settings.rng_seed
        )
        likelihood_component = posterior.GaussianLLFromPTOMap(
            pto_model, likelihood_settings.data, likelihood_settings.covariance
        )
        model_wrapper = posterior.LogPosterior(prior_component, likelihood_component)
        models = [partial(model_wrapper, config=mesh_file_name) for mesh_file_name in configs]

        return models, prior_component

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def set_up_sampler_components(proposal_settings, accept_rate_settings):
        ground_proposal = mcmc.RandomWalkProposal(
            proposal_settings.step_width,
            proposal_settings.covariance,
            proposal_settings.rng_seed,
        )
        accept_rate_estimator = mcmc.MLAcceptRateEstimator(
            accept_rate_settings.initial_guess,
            accept_rate_settings.update_parameter,
        )

        return ground_proposal, accept_rate_estimator
