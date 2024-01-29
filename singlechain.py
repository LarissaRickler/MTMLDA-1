import numpy as np
import umbridge

from proposals.proposals import RandomWalkProposal, RandomWalkProposalSettings
from src.mtmlda.mlmcmc import MLAcceptRateEstimator, MLAcceptRateEstimatorSettings
from src.mtmlda.sampler import MTMLDASampler, SamplerSetupSettings, SamplerRunSettings


proposal_settings = RandomWalkProposalSettings(
    step_width=0.01,
    covariance=np.identity(2),
    rng_seed=0
)

accept_rate_settings = MLAcceptRateEstimatorSettings(
    initial_guess=[0.5, 0.7, 0.8],
    update_parameter=0.01
)

sampler_setup_settings = SamplerSetupSettings(
    num_levels=3,
    subsampling_rates=[5, 3, -1],
    rng_seed=0,
    do_printing=True,
    logfile="singlechain.log"
)

sampler_run_settings = SamplerRunSettings(
    num_samples=5,
    initial_state=np.array([4, 4]),
    num_threads=8,
    rng_seed=0
)

models = [
    umbridge.HTTPModel("http://localhost:4243", "posterior_coarse"),
    umbridge.HTTPModel("http://localhost:4243", "posterior_intermediate"),
    umbridge.HTTPModel("http://localhost:4243", "posterior_fine"),
]


def main():
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

    mcmc_chain = sampler.run(sampler_run_settings)


if __name__ == "__main__":
    main()
