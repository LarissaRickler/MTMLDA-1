from pathlib import Path

import numpy as np

from components import general_settings
from . import builder



# ==================================================================================================
parallel_run_settings = general_settings.ParallelRunSettings(
    num_chains=1,
    result_directory_path=Path("results_seissol_zihua_4D"),
    chain_file_stem=Path("chain"),
    rng_state_save_file_stem=None,
    rng_state_load_file_stem=None,
    overwrite_results=True,
)

sampler_setup_settings = general_settings.SamplerSetupSettings(
    num_levels=2,
    subsampling_rates=[5, -1],
    max_tree_height=50,
    underflow_threshold=-1e4,
    rng_seed_mltree=None,
    rng_seed_node_init=None,
    mltree_path=None,
)

sampler_run_settings = general_settings.SamplerRunSettings(
    num_samples=2,
    initial_state=None,
    num_threads=1,
    print_interval=1,
    tree_render_interval=1,
)

logger_settings = general_settings.LoggerSettings(
    do_printing=True,
    logfile_path=Path("results_seissol_zihua_4D") / Path("mtmlda"),
    debugfile_path=Path("results_seissol_zihua_4D") / Path("mtmlda_debug"),
    write_mode="w",
)

# --------------------------------------------------------------------------------------------------
inverse_problem_settings = builder.InverseProblemSettings(
    prior_intervals=np.array([[500, 2000], [1, 20], [20e9, 30e9], [20e9, 30e9]]),
    prior_rng_seed=None,
    likelihood_data_dir=Path("seissol/zihua_4D/data"),
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
