from pathlib import Path

import numpy as np

import src.mtmlda.sampler as sampler


# ==================================================================================================
class Settings:

    class model_settings:
        configs = ({"meshFile": "model_0p1Hz"}, {"meshFile": "model_0p3Hz"})
        address = "http://localhost:4242"
        name = "forward"

    class prior_settings:
        parameter_intervals = np.array([[500, 2000], [1, 20], [20e9, 30e9], [20e9, 30e9]])
        rng_seed = None

    class likelihood_settings:
        data = np.array([0, 0, 0, 0])
        covariance = np.identity(4)

    class proposal_settings:
        step_width = 0.1
        covariance = np.diag((np.square(1500), np.square(19), np.square(10e9), np.square(10e9)))
        rng_seed = None

    class accept_rate_settings:
        initial_guess = [0.5, 0.7]
        update_parameter = 0.01

    class run_settings:
        num_chains = 4
        result_directory_path = Path("results_example_01")
        chain_file_stem = Path("chain")
        rng_state_save_file_stem = None
        rng_state_load_file_stem = None
        overwrite_results = True

    sampler_setup_settings = sampler.SamplerSetupSettings(
        num_levels=2,
        subsampling_rates=[5, -1],
        max_tree_height=50,
        rng_seed_mltree=None,
        rng_seed_node_init=None,
        do_printing=True,
        mltree_path=Path("results_example_01") / Path("mltree"),
        logfile_path=Path("results_example_01") / Path("mtmlda.log"),
        write_mode="w",
    )

    sampler_run_settings = sampler.SamplerRunSettings(
        num_samples=5000,
        initial_state=None,
        num_threads=8,
        print_interval=100,
        tree_render_interval=100,
    )
