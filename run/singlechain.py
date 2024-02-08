import os
import sys
from pathlib import Path

sys.path.append(str(Path("../src/").resolve()))

import numpy as np

from mtmlda.mcmc import RandomWalkProposal, MLAcceptRateEstimator
from mtmlda.sampler import MTMLDASampler

import settings


# ==================================================================================================
class result_settings:
    result_directory_path = Path("../results") / Path("chain")
    overwrite_results = True


# ==================================================================================================
def set_up_sampler(
    proposal_settings, accept_rate_settings, sampler_setup_settings, models
):
    ground_proposal = RandomWalkProposal(
        proposal_settings.step_width,
        proposal_settings.covariance,
        proposal_settings.rng_seed,
    )
    accept_rate_estimator = MLAcceptRateEstimator(
        accept_rate_settings.initial_guess,
        accept_rate_settings.update_parameter,
    )
    sampler = MTMLDASampler(
        sampler_setup_settings,
        models,
        accept_rate_estimator,
        ground_proposal,
    )
    return sampler


def main():
    os.makedirs(
        result_settings.result_directory_path.parent, exist_ok=result_settings.overwrite_results
    )
    sampler = set_up_sampler(
        settings.proposal_settings,
        settings.accept_rate_settings,
        settings.sampler_setup_settings,
        settings.models,
    )
    mcmc_chain = sampler.run(settings.sampler_run_settings)
    np.save(result_settings.result_directory_path, mcmc_chain)


if __name__ == "__main__":
    main()
