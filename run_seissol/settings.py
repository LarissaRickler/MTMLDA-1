from pathlib import Path

import numpy as np
import umbridge as ub

import src.mtmlda.sampler as sampler
import models.posterior_pto_wrapper as wrapper


# ==================================================================================================
class prior_settings:
    parameter_intervals = np.array([[500, 5000], [1e5, 1e6], [5, 50], [5, 50]])


class likelihood_settings:
    space_data, space_variance = np.load("data/space_data.npz").values()
    time_data, time_variance = np.load("data/time_data.npz").values()
    data = np.concatenate((space_data, time_data))
    space_covariance = np.diag(space_variance)
    time_covariance = np.diag(time_variance)
    covariance = np.block([[space_covariance, np.zeros((space_data.size, time_data.size))],
                           [np.zeros((time_data.size, space_data.size)), time_covariance]])

class proposal_settings:
    step_width = 0.1
    covariance = np.identity(4)
    rng_seed = 0


class accept_rate_settings:
    initial_guess = [0.5, 0.7]
    update_parameter = 0.01


sampler_setup_settings = sampler.SamplerSetupSettings(
    num_levels=2,
    subsampling_rates=[5, -1],
    max_tree_height=50,
    rng_seed=0,
    do_printing=True,
    mltree_path=Path("results") / Path("mltree"),
    logfile_path=Path("results") / Path("mtmlda.log"),
    write_mode="w",
)

sampler_run_settings = sampler.SamplerRunSettings(
    num_samples=2,
    initial_state=np.array([2000, 5e5, 32, 17]),
    num_threads=8,
    rng_seed=0,
    print_interval=1,
    tree_render_interval=1,
)


# ==================================================================================================
pto_model_data = {
    0: ("http://localhost:4243", "parameter_to_observable_map_coarse"),
    1: ("http://localhost:4243", "parameter_to_observable_map_fine"),
}

pto_models = [ub.HTTPModel(*pto_model_data[0]), ub.HTTPModel(*pto_model_data[1])]
priors = [wrapper.UninformLogPrior(prior_settings.parameter_intervals),] * len(pto_models)
likelihoods = [
    wrapper.GaussianLogLikelihood(
        pto_model, likelihood_settings.data, likelihood_settings.covariance
    )
    for pto_model in pto_models
]

models = [wrapper.LogPosterior(prior, likelihood) for prior, likelihood in zip(priors, likelihoods)]
