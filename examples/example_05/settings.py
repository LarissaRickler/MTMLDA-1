import datetime
from pathlib import Path

import numpy as np

from components import general_settings

from . import builder

# ==================================================================================================
timestamp = datetime.datetime.now()
timestamp = f"{timestamp.year:04d}{timestamp.month:02d}{timestamp.day:02d}" \
            f"_{timestamp.hour:02d}{timestamp.minute:02d}{timestamp.second:02d}"
result_directory = f"{timestamp}_example_05"


parallel_run_settings = general_settings.ParallelRunSettings(
    num_chains=1,
    chain_save_path=Path(f"{result_directory}/chain"),
    chain_load_path=None,
    node_save_path=None,
    node_load_path=None,
    rng_state_save_path=None,
    rng_state_load_path=None,
    overwrite_chain=True,
    overwrite_node=True,
    overwrite_rng_states=True,
)

sampler_setup_settings = general_settings.SamplerSetupSettings(
    num_levels=3,
    subsampling_rates=[30, 3, -1],
    max_tree_height=50,
    underflow_threshold=-1000,
    rng_seed_mltree=1,
    rng_seed_node_init=2,
    mltree_path=Path(f"{result_directory}/mltree"),
)

sampler_run_settings = general_settings.SamplerRunSettings(
    num_samples=3,
    initial_state=None,
    initial_node=None,
    num_threads=6,
    print_interval=10,
)

logger_settings = general_settings.LoggerSettings(
    do_printing=True,
    logfile_path=Path(f"{result_directory}/mtmlda"),
    debugfile_path=Path(f"{result_directory}/mtmlda_debug"),
    write_mode="w",
)

# --------------------------------------------------------------------------------------------------
inverse_problem_settings = builder.InverseProblemSettings(
    ub_model_configs=({"level": 0}, {"level": 1}, {"level": 2}),
    ub_model_address="http://localhost:4242",
    ub_model_name="banana_posterior",
)

sampler_component_settings = builder.SamplerComponentSettings(
    proposal_step_width=1,
    proposal_covariance=1 * np.identity(2),
    proposal_rng_seed=3,
    accept_rates_initial_guess=[0.5, 0.7, 0.8],
    accept_rates_update_parameter=0.01,
)

initial_state_settings = builder.InitialStateSettings(
    rng_seed_init=4,
    mean_init=np.array([0, 0]),
    covariance_init=np.identity(2),
)
