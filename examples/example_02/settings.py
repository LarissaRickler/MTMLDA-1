from pathlib import Path

import numpy as np
from components import general_settings

from . import builder

# ==================================================================================================
parallel_run_settings = general_settings.ParallelRunSettings(
    num_chains=4,
    result_directory_path=Path("results_example_02"),
    chain_file_stem=Path("chain"),
    rng_state_save_file_stem=None,
    rng_state_load_file_stem=None,
    overwrite_results=True,
)

sampler_setup_settings = general_settings.SamplerSetupSettings(
    num_levels=3,
    subsampling_rates=[5, 3, -1],
    max_tree_height=50,
    underflow_threshold=-1000,
    rng_seed_mltree=1,
    rng_seed_node_init=2,
    mltree_path=None,
)

sampler_run_settings = general_settings.SamplerRunSettings(
    num_samples=2500,
    initial_state=None,
    num_threads=8,
    print_interval=100,
    tree_render_interval=100,
)

logger_settings = general_settings.LoggerSettings(
    do_printing=True,
    logfile_path=Path("results_example_02") / Path("mtmlda"),
    debugfile_path=None,
    write_mode="w",
)

# --------------------------------------------------------------------------------------------------
inverse_problem_settings = builder.InverseProblemSettings(
    ub_model_address="http://localhost:4242",
    ub_model_names=[
        "banana_posterior_coarse",
        "banana_posterior_intermediate",
        "banana_posterior_fine",
    ],
)

sampler_component_settings = builder.SamplerComponentSettings(
    proposal_step_width=0.1,
    proposal_covariance=2 * np.identity(2),
    proposal_rng_seed=3,
    accept_rates_initial_guess=[0.5, 0.7, 0.8],
    accept_rates_update_parameter=0.01,
)

initial_state_settings = builder.InitialStateSettings(
    initial_states=[
        np.array((-0.1, -0.1)),
        np.array((-0.1, +0.1)),
        np.array((+0.1, -0.1)),
        np.array((+0.1, +0.1)),
    ]
)
