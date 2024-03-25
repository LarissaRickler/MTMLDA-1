from pathlib import Path

import numpy as np
from components import general_settings

from . import builder

# ==================================================================================================
parallel_run_settings = general_settings.ParallelRunSettings(
    num_chains=4,
    result_directory_path=Path("results_example_03"),
    chain_file_stem=Path("chain"),
    rng_state_save_file_stem=None,
    rng_state_load_file_stem=None,
    overwrite_results=True,
)

sampler_setup_settings = general_settings.SamplerSetupSettings(
    num_levels=2,
    subsampling_rates=[5, -1],
    max_tree_height=50,
    underflow_threshold=-1000,
    rng_seed_mltree=None,
    rng_seed_node_init=None,
    do_printing=True,
    mltree_path=Path("results_example_03") / Path("mltree"),
    logfile_path=Path("results_example_03") / Path("mtmlda.log"),
    write_mode="w",
)

sampler_run_settings = general_settings.SamplerRunSettings(
    num_samples=2500,
    initial_state=None,
    num_threads=8,
    print_interval=100,
    tree_render_interval=100,
)

# --------------------------------------------------------------------------------------------------
inverse_problem_settings = builder.InverseProblemSettings(
    prior_intervals=np.array([[500, 2000], [1, 20], [20e9, 30e9], [20e9, 30e9]]),
    prior_rng_seed=None,
    likelihood_data=np.array([0, 0, 0, 0]),
    likelihood_covariance=np.identity(4),
    ub_model_configs=({"meshFile": "model_0p1Hz"}, {"meshFile": "model_0p3Hz"}),
    ub_model_address="http://localhost:4242",
    ub_model_name="forward",
)

sampler_component_settings = builder.SamplerComponentSettings(
    proposal_step_width=0.1,
    proposal_covariance=np.diag((np.square(1500), np.square(19), np.square(10e9), np.square(10e9))),
    proposal_rng_seed=None,
    accept_rates_initial_guess=[0.5, 0.7],
    accept_rates_update_parameter=0.01,
)

initial_state_settings = builder.InitialStateSettings()
