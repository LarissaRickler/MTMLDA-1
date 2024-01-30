import os
from pathlib import Path
import numpy as np
import umbridge

from proposals.proposals import RandomWalkProposal
from src.mtmlda.mlmcmc import MLAcceptRateEstimator
from src.mtmlda.sampler import MTMLDASampler, SamplerSetupSettings, SamplerRunSettings


# ==================================================================================================
class result_settings:
    result_directory_path = Path("results") / Path("chain")
    overwrite_results = True

class proposal_settings:
    step_width = (0.1,)
    covariance = (np.identity(2),)
    rng_seed = 0

class accept_rate_settings:
    initial_guess = ([0.5, 0.7, 0.8],)
    update_parameter = 0.01

sampler_setup_settings = SamplerSetupSettings(
    num_levels=3,
    subsampling_rates=[5, 3, -1],
    rng_seed=0,
    do_printing=True,
    mltree_path=Path("results") / Path("mltree"),
    logfile_path=Path("results") / Path("singlechain.log"),
    write_mode="w",
)

sampler_run_settings = SamplerRunSettings(
    num_samples=100,
    initial_state=np.array([4, 4]),
    num_threads=8,
    rng_seed=0,
    print_interval=10
)

models = [
    umbridge.HTTPModel("http://localhost:4243", "banana_posterior_coarse"),
    umbridge.HTTPModel("http://localhost:4243", "banana_posterior_intermediate"),
    umbridge.HTTPModel("http://localhost:4243", "banana_posterior_fine"),
]


# ==================================================================================================
def set_up_sampler():
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
    sampler = set_up_sampler()
    mcmc_chain = sampler.run(sampler_run_settings)
    np.save(result_settings.result_directory_path, mcmc_chain)


if __name__ == "__main__":
    main()
