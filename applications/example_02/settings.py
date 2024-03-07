from pathlib import Path

import numpy as np

from components import general_settings
from . import builder

# ==================================================================================================
parallel_run_settings = general_settings.ParallelRunSettings(
    num_chains=1,
    result_directory_path=Path("results_example_02"),
    chain_file_stem=Path("chain"),
    rng_state_save_file_stem=Path("rng_states"),
    rng_state_load_file_stem=None,
    overwrite_results=True,
)

sampler_setup_settings = general_settings.SamplerSetupSettings(
    num_levels=2,
    subsampling_rates=[5, -1],
    max_tree_height=50,
    rng_seed_mltree=None,
    rng_seed_node_init=None,
    do_printing=True,
    mltree_path=Path("results_example_02") / Path("mltree"),
    logfile_path=Path("results_example_02") / Path("mtmlda.log"),
    write_mode="w",
)

sampler_run_settings = general_settings.SamplerRunSettings(
    num_samples=20,
    initial_state=None,
    num_threads=3,
    print_interval=2,
    tree_render_interval=2,
)

# --------------------------------------------------------------------------------------------------
inverse_problem_settings = builder.InverseProblemSettings(
    prior_mean=np.array((0,)),
    prior_covariance=np.identity(1),
    ub_model_configs=({"order": 4}, {"order": 5}),
    ub_model_address="http://localhost:4242",
    ub_model_name="forward",
)

sampler_component_settings = builder.SamplerComponentSettings(
    proposal_step_width=0.1,
    proposal_covariance=np.identity(1),
    proposal_rng_seed=None,
    accept_rates_initial_guess=[0.5, 0.7],
    accept_rates_update_parameter=0.01,
)

initial_state_settings = builder.InitialStateSettings()
