from pathlib import Path

import numpy as np

from mtmlda.core import logging, sampling
from mtmlda.run import postprocessor, runner

from . import builder

# ==================================================================================================
result_directory = "../results_example_01"

parallel_run_settings = runner.ParallelRunSettings(
    num_chains=2,
    chain_save_path=Path(f"{result_directory}/chain"),
    chain_load_path=None,
    rng_state_save_path=None,
    rng_state_load_path=None,
    overwrite_chain=True,
    overwrite_rng_states=True,
)

sampler_setup_settings = sampling.SamplerSetupSettings(
    num_levels=2,
    subsampling_rates=[5, -1],
    max_tree_height=50,
    underflow_threshold=-1000,
    rng_seed_mltree=1,
    rng_seed_node_init=2,
    mltree_path=None,
)

sampler_run_settings = sampling.SamplerRunSettings(
    num_samples=10,
    initial_state=None,
    num_threads=2,
    print_interval=1,
)

logger_settings = logging.LoggerSettings(
    do_printing=True,
    logfile_path=Path(f"{result_directory}") / Path("mtmlda"),
    debugfile_path=Path(f"{result_directory}/mtmlda_debug"),
    write_mode="w",
)

# --------------------------------------------------------------------------------------------------
inverse_problem_settings = builder.InverseProblemSettings(
    prior_intervals=np.array([[500, 2000], [1, 20], [20e9, 30e9], [20e9, 30e9]]),
    prior_rng_seed=3,
    likelihood_data=np.array([0, 0, 0, 0]),
    likelihood_covariance=np.identity(4),
    ub_model_configs=({"level": "0"}, {"level": "1"}),
    ub_model_address="http://localhost:4242",
    ub_model_name="banana_pto",
)

sampler_component_settings = builder.SamplerComponentSettings(
    proposal_step_width=0.1,
    proposal_covariance=np.diag((np.square(1500), np.square(19), np.square(10e9), np.square(10e9))),
    proposal_rng_seed=4,
    accept_rates_initial_guess=[0.5, 0.7],
    accept_rates_update_parameter=0.01,
)

initial_state_settings = builder.InitialStateSettings()

# --------------------------------------------------------------------------------------------------
postprocessor_settings = postprocessor.PostprocessorSettings(
    chain_directory=Path(result_directory),
    tree_directory=None,
    output_data_directory=Path(result_directory),
    visualization_directory=Path(result_directory),
    acf_max_lag=9,
)
