import functools
import multiprocessing
import os
from pathlib import Path

import numpy as np
import umbridge

from proposals.proposals import RandomWalkProposal
from src.mtmlda.mlmcmc import MLAcceptRateEstimator
from src.mtmlda.sampler import MTMLDASampler
import settings


# ==================================================================================================
class run_settings:
    num_chains = 4
    result_directory_path = Path("results")
    overwrite_results = True


# ==================================================================================================
def execute_mtmlda_run(process_id):
    _modify_process_dependent_settings(process_id)
    sampler = _set_up_sampler()
    mcmc_chain = sampler.run(settings.sampler_run_settings)
    _save_trace(process_id, mcmc_chain)

def _modify_process_dependent_settings(process_id):
    settings.proposal_settings.rng_seed = process_id
    settings.sampler_setup_settings.rng_seed = process_id
    settings.sampler_run_settings.rng_seed = process_id
    settings.sampler_setup_settings.mltree_path = None

    if process_id != 0:
        settings.sampler_setup_settings.do_printing = False
        settings.sampler_setup_settings.logfile_path = None

    process_rng = np.random.default_rng(process_id)
    perturbation = process_rng.normal(0, 0.1, size=settings.sampler_run_settings.initial_state.size)
    settings.sampler_run_settings.initial_state = perturbation

def _set_up_sampler():
    ground_proposal = RandomWalkProposal(
        settings.proposal_settings.step_width,
        settings.proposal_settings.covariance,
        settings.proposal_settings.rng_seed,
    )
    accept_rate_estimator = MLAcceptRateEstimator(
        settings.accept_rate_settings.initial_guess,
        settings.accept_rate_settings.update_parameter,
    )
    sampler = MTMLDASampler(
        settings.sampler_setup_settings,
        settings.models,
        accept_rate_estimator,
        ground_proposal,
    )
    return sampler

def _save_trace(process_id, mcmc_trace):
    os.makedirs(run_settings.result_directory_path, exist_ok=run_settings.overwrite_results)
    file_name = (
        run_settings.result_directory_path / Path(f"chain_{process_id}")
    )
    np.save(file_name, mcmc_trace)


# --------------------------------------------------------------------------------------------------
def main():
    process_ids = range(run_settings.num_chains)
    with multiprocessing.Pool(processes=run_settings.num_chains) as process_pool:
        process_pool.map(execute_mtmlda_run, process_ids)


if __name__ == "__main__":
    main()
