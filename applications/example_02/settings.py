from dataclasses import dataclass
from pathlib import Path

import numpy as np

import components.settings as settings


# ==================================================================================================
@dataclass
class InverseProblemSettings(settings.InverseProblemSettings):
    prior_mean: np.ndarray
    prior_covariance: np.ndarray
    ub_model_configs: dict[str, str]
    ub_model_address: str
    ub_model_name: str


@dataclass
class SamplerComponentSettings(settings.SamplerComponentSettings):
    proposal_step_width: float
    proposal_covariance: np.ndarray
    proposal_rng_seed: int
    accept_rates_initial_guess: list[float]
    accept_rates_update_parameter = float

@dataclass
class InitialStateSettings(settings.InitialStateSettings):
    pass


# ==================================================================================================
parallel_run_settings = settings.ParallelRunSettings(
    num_chains=4,
    result_directory_path=Path("results_example_02"),
    chain_file_stem=Path("chain"),
    rng_state_save_file_stem=Path("rng_states"),
    rng_state_load_file_stem=None,
    overwrite_results=True
)

sampler_setup_settings = settings.SamplerSetupSettings(
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

sampler_run_settings = settings.SamplerRunSettings(
    num_samples=5000,
    initial_state=None,
    num_threads=8,
    print_interval=100,
    tree_render_interval=100,
)

# --------------------------------------------------------------------------------------------------
inverse_problem_settings = InverseProblemSettings(
    prior_mean=np.array((0,)),
    prior_covariance=np.identity(1),
    ub_model_configs=({"meshFile": "model_0p1Hz"}, {"meshFile": "model_0p3Hz"}),
    ub_model_address="http://localhost:4242",
    ub_model_name="forward",
)

sampler_component_settings = SamplerComponentSettings(
    proposal_step_width=0.1,
    proposal_covariance=np.identity(1),
    proposal_rng_seed=None,
    accept_rates_initial_guess=[0.5, 0.7],
    accept_rates_update_parameter=0.01
)

initial_state_settings = InitialStateSettings()
