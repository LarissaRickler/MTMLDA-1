import os
import time
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
    parameter_intervals = np.array([[500, 5000], [1e5, 1e6], [5, 50], [5, 50]])


class likelihood_settings:
    space_data, space_variance = np.load("seissol/data/space_data.npz").values()
    time_data, time_variance = np.load("seissol/data/time_data.npz").values()
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
    initial_state=np.array([2000, 5e5, 32, 17]),
    num_threads=8,
    rng_seed_node_init=0,
    print_interval=1,
    tree_render_interval=1,
)

# ==================================================================================================
### change me to the actual mesh file names
pto_model_config = ("coarse", "fine")
server_available = False
while not server_available:
        try:
                pto_model = ub.HTTPModel("https://localhost:4242", "forward")
                pto_model = ub.HTTPModel("http://localhost:4243", "forward")
                print("Server available")
                server_available = True
        except:
                print("Server not available")
                time.sleep(10)


prior = wrapper.UninformLogPrior(prior_settings.parameter_intervals)
likelihood = wrapper.GaussianLogLikelihood(
    pto_model, likelihood_settings.data, likelihood_settings.covariance
)
model_wrapper = wrapper.LogPosterior(prior, likelihood)
models = [partial(model_wrapper, config=mesh_file_name) for mesh_file_name in pto_model_config]
