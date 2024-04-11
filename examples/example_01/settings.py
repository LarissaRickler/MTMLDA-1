from pathlib import Path

import numpy as np
from components import general_settings

from . import builder


# ==================================================================================================
parallel_run_settings = general_settings.ParallelRunSettings(
    num_chains=4,
    chain_save_path=Path("results_example_01/chain"),
    chain_load_path=None,
    rng_state_save_path=None,
    rng_state_load_path=None,
    overwrite_chain=True,
    overwrite_rng_states=True,
)

sampler_setup_settings = general_settings.SamplerSetupSettings(
    num_levels=3,
    subsampling_rates=[5, 3, -1],
    max_tree_height=50,
    underflow_threshold=-1000,
    rng_seed_mltree=1,
    rng_seed_node_init=2,
    mltree_path=Path("results_example_01/mltree"),
)

sampler_run_settings = general_settings.SamplerRunSettings(
    num_samples=2,
    initial_state=None,
    num_threads=8,
    print_interval=1,
    tree_render_interval=1,
)

logger_settings = general_settings.LoggerSettings(
    do_printing=True,
    logfile_path=Path("results_example_01/mtmlda"),
    debugfile_path=Path("results_example_01/mtmlda_debug"),
    write_mode="w",
)

# --------------------------------------------------------------------------------------------------
inverse_problem_settings = builder.InverseProblemSettings(
    ub_model_address="http://localhost:4242",
    ub_model_names=[
        "gaussian_posterior_coarse",
        "gaussian_posterior_intermediate",
        "gaussian_posterior_fine",
    ],
)

sampler_component_settings = builder.SamplerComponentSettings(
    proposal_step_width=0.1,
    proposal_covariance=0.1 * np.identity(2),
    proposal_rng_seed=3,
    accept_rates_initial_guess=[0.5, 0.7, 0.8],
    accept_rates_update_parameter=0.01,
)

initial_state_settings = builder.InitialStateSettings(
    initial_states=[
        np.array((-10, -0.1)),
        np.array((-0.1, +0.1)),
        np.array((+0.1, -0.1)),
        np.array((+0.1, +0.1)),
    ]
)
