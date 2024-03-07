from pathlib import Path

import numpy as np

import src.mtmlda.sampler as sampler


# ==================================================================================================
class Settings:

    class model_settings:
        configs = ({"order": 4}, {"order": 5})
        address = "http://localhost:4242"
        name = "forward"

    class prior_settings:
        mean = np.array((5000000,))
        covariance = 1e12*np.identity(1)
        rng_seed = None

    class likelihood_settings:
        pass

    class proposal_settings:
        step_width = 0.1
        covariance = 1e12*np.identity(1)
        rng_seed = None

    class accept_rate_settings:
        initial_guess = [0.5, 0.7]
        update_parameter = 0.01

    class run_settings:
        num_chains = 1
        result_directory_path = Path("results_seissol_sebastian")
        chain_file_stem = Path("chain")
        rng_state_save_file_stem = Path("rng_states")
        rng_state_load_file_stem = None
        overwrite_results = True

    sampler_setup_settings = sampler.SamplerSetupSettings(
        num_levels=2,
        subsampling_rates=[1, -1],
        max_tree_height=50,
        rng_seed_mltree=None,
        rng_seed_node_init=None,
        do_printing=True,
        mltree_path=Path("results_seissol_sebastian") / Path("mltree"),
        logfile_path=Path("results_seissol_sebastian") / Path("mtmlda.log"),
        write_mode="w",
    )

    sampler_run_settings = sampler.SamplerRunSettings(
        num_samples=2,
        initial_state=None,
        num_threads=3,
        print_interval=1,
        tree_render_interval=1,
    )
