import os
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path("../").resolve()))

import numpy as np

import src.mtmlda.mcmc as mcmc
import src.mtmlda.sampler as sampler
import settings


# ==================================================================================================
class run_settings:
    result_directory_path = Path("results") / Path("chain")
    overwrite_results = True
    rng_state_load_file = Path("results") / Path("rng_states.pkl")
    rng_state_save_file = Path("results") / Path("rng_states.pkl")


# ==================================================================================================
def set_up_sampler(
    run_settings, proposal_settings, accept_rate_settings, sampler_setup_settings, models
):
    ground_proposal = mcmc.RandomWalkProposal(
        proposal_settings.step_width,
        proposal_settings.covariance,
        proposal_settings.rng_seed,
    )
    accept_rate_estimator = mcmc.MLAcceptRateEstimator(
        accept_rate_settings.initial_guess,
        accept_rate_settings.update_parameter,
    )
    mtmlda_sampler = sampler.MTMLDASampler(
        sampler_setup_settings,
        models,
        accept_rate_estimator,
        ground_proposal,
    )
    if run_settings.rng_state_load_file is not None:
        with run_settings.rng_state_load_file.open("rb") as rng_state_file:
            rng_states = pickle.load(rng_state_file)
        mtmlda_sampler.set_rngs(rng_states)
    return mtmlda_sampler


def main():
    os.makedirs(
        run_settings.result_directory_path.parent, exist_ok=run_settings.overwrite_results
    )
    mtmlda_sampler = set_up_sampler(
        run_settings,
        settings.proposal_settings,
        settings.accept_rate_settings,
        settings.sampler_setup_settings,
        settings.models,
    )
    mcmc_chain = mtmlda_sampler.run(settings.sampler_run_settings)
    np.save(run_settings.result_directory_path, mcmc_chain)
    if run_settings.rng_state_save_file is not None:
        with run_settings.rng_state_save_file.open("wb") as rng_state_file:
            pickle.dump(mtmlda_sampler.get_rngs(), rng_state_file)


if __name__ == "__main__":
    main()
