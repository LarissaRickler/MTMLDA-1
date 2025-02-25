from pathlib import Path

import numpy as np

from mtmlda.core import logging, sampling
from mtmlda.run import postprocessor, runner

from . import builder

# ==================================================================================================
result_directory = "../results_example_03"

parallel_run_settings = runner.ParallelRunSettings(
    num_chains=4,
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

sampler_setup_settings = sampling.SamplerSetupSettings(
    num_levels=3,
    subsampling_rates=[30, 3, -1],
    max_tree_height=50,
    underflow_threshold=-1000,
    rng_seed_mltree=1,
    rng_seed_node_init=2,
    mltree_path=None,
)

sampler_run_settings = sampling.SamplerRunSettings(
    num_samples=1000,
    initial_state=None,
    initial_node=None,
    num_threads=2,
    print_interval=50,
)

logger_settings = logging.LoggerSettings(
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

# --------------------------------------------------------------------------------------------------
postprocessor_settings = postprocessor.PostprocessorSettings(
    chain_directory=Path(result_directory),
    tree_directory=None,
    output_data_directory=Path(result_directory),
    visualization_directory=Path(result_directory),
    acf_max_lag=100,
)
