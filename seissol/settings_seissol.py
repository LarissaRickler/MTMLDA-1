from functools import partial
from pathlib import Path

import models.posterior_pto_wrapper as wrapper
import numpy as np
import src.mtmlda.sampler as sampler
import umbridge as ub


# ==================================================================================================
class run_settings:
    num_chains = 4
    result_directory_path = Path("results")
    chain_file_stem = Path("chain")
    rng_state_save_file_stem = Path("rng_states")
    rng_state_load_file_stem = None
    overwrite_results = True


class prior_settings:
    parameter_intervals = np.array([[500, 2000], [1, 20], [20e9, 30e9], [20e9, 30e9]])


class likelihood_settings:
    data_directory = Path("seissol").resolve() / Path("data")
    space_data, space_variance = np.load(data_directory / Path("space_data.npz")).values()
    time_data, time_variance = np.load(data_directory / Path("time_data.npz")).values()
    data = np.concatenate((space_data, time_data))
    space_covariance = np.diag(space_variance)
    time_covariance = np.diag(time_variance)
    covariance = np.block(
        [
            [space_covariance, np.zeros((space_data.size, time_data.size))],
            [np.zeros((time_data.size, space_data.size)), time_covariance],
        ]
    )


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
    rng_seed_mltree=0,
    do_printing=True,
    mltree_path=Path("results") / Path("mltree"),
    logfile_path=Path("results") / Path("mtmlda.log"),
    write_mode="w",
)

sampler_run_settings = sampler.SamplerRunSettings(
    num_samples=2,
    initial_state=np.array([1000, 10, 28518750000, 24637500000]),
    num_threads=8,
    rng_seed_node_init=0,
    print_interval=1,
    tree_render_interval=1,
)


# ==================================================================================================
pto_model_config = ("coarse", "fine")
pto_model = ub.HTTPModel("http://localhost:4243", "forward")

prior = wrapper.UninformLogPrior(prior_settings.parameter_intervals)
likelihood = wrapper.GaussianLogLikelihood(
    pto_model, likelihood_settings.data, likelihood_settings.covariance
)
model_wrapper = wrapper.LogPosterior(prior, likelihood)
models = [partial(model_wrapper, config=mesh_file_name) for mesh_file_name in pto_model_config]
