from pathlib import Path

import numpy as np
import umbridge as ub

import mtmlda.sampler as sampler


# ==================================================================================================
class proposal_settings:
    step_width = 0.1
    covariance = np.identity(2)
    rng_seed = 0

class accept_rate_settings:
    initial_guess = [0.5, 0.7, 0.8]
    update_parameter = 0.01

sampler_setup_settings = sampler.SamplerSetupSettings(
    num_levels=3,
    subsampling_rates=[5, 3, -1],
    max_tree_height=50,
    rng_seed=0,
    do_printing=True,
    mltree_path=Path("../results") / Path("mltree"),
    logfile_path=Path("../results") / Path("mtmlda.log"),
    write_mode="w",
)

sampler_run_settings = sampler.SamplerRunSettings(
    num_samples=100,
    initial_state=np.array([0, 0]),
    num_threads=8,
    rng_seed=0,
    print_interval=10,
    tree_render_interval = 10
)


# ==================================================================================================
models = [
    ub.HTTPModel("http://localhost:4243", "gauss_posterior_coarse"),
    ub.HTTPModel("http://localhost:4243", "gauss_posterior_intermediate"),
    ub.HTTPModel("http://localhost:4243", "gauss_posterior_fine"),
]