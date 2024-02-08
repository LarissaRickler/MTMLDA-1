from pathlib import Path

import numpy as np
import umbridge

from mtmlda.sampler import SamplerSetupSettings, SamplerRunSettings


# ==================================================================================================
class proposal_settings:
    step_width = 0.1
    covariance = np.identity(2)
    rng_seed = 0

class accept_rate_settings:
    initial_guess = [0.5, 0.7, 0.8]
    update_parameter = 0.01

sampler_setup_settings = SamplerSetupSettings(
    num_levels=3,
    subsampling_rates=[5, 3, -1],
    max_tree_height=50,
    rng_seed=0,
    do_printing=True,
    mltree_path=Path("../results") / Path("mltree"),
    logfile_path=Path("../results") / Path("mtmlda.log"),
    write_mode="w",
)

sampler_run_settings = SamplerRunSettings(
    num_samples=100,
    initial_state=np.array([0, 0]),
    num_threads=8,
    rng_seed=0,
    print_interval=10,
    tree_render_interval = 10
)


# ==================================================================================================
models = [
    umbridge.HTTPModel("http://localhost:4243", "gauss_posterior_coarse"),
    umbridge.HTTPModel("http://localhost:4243", "gauss_posterior_intermediate"),
    umbridge.HTTPModel("http://localhost:4243", "gauss_posterior_fine"),
]