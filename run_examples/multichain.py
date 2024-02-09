import multiprocessing
import os
import sys
from functools import partial
from pathlib import Path

sys.path.append(str(Path("../src/").resolve()))

import numpy as np

import mtmlda.mcmc as mcmc
import mtmlda.sampler as sampler
import settings


# ==================================================================================================
class run_settings:
    num_chains = 4
    result_directory_path = Path("results")
    overwrite_results = True


# ==================================================================================================
def execute_mtmlda_run(
    process_id,
    run_settings,
    proposal_settings,
    accept_rate_settings,
    sampler_setup_settings,
    sampler_run_settings,
    models,
):
    _modify_process_dependent_settings(
        process_id, proposal_settings, sampler_setup_settings, sampler_run_settings
    )
    mtmlda_sampler = _set_up_sampler(
        proposal_settings, accept_rate_settings, sampler_setup_settings, models
    )
    mcmc_chain = mtmlda_sampler.run(sampler_run_settings)
    _save_trace(process_id, run_settings, mcmc_chain)


def _modify_process_dependent_settings(
    process_id: int, proposal_settings, sampler_setup_settings, sampler_run_settings
) -> None:
    proposal_settings.rng_seed = process_id
    sampler_setup_settings.rng_seed = process_id
    sampler_run_settings.rng_seed = process_id
    sampler_setup_settings.mltree_path = None

    if process_id != 0:
        sampler_setup_settings.do_printing = False
        sampler_setup_settings.logfile_path = None

    process_rng = np.random.default_rng(process_id)
    perturbation = process_rng.normal(0, 0.1, size=sampler_run_settings.initial_state.size)
    sampler_run_settings.initial_state = perturbation


def _set_up_sampler(proposal_settings, accept_rate_settings, sampler_setup_settings, models):
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
    return mtmlda_sampler


def _save_trace(process_id, run_settings, mcmc_trace) -> None:
    os.makedirs(run_settings.result_directory_path, exist_ok=run_settings.overwrite_results)
    file_name = run_settings.result_directory_path / Path(f"chain_{process_id}")
    np.save(file_name, mcmc_trace)


# --------------------------------------------------------------------------------------------------
def main() -> None:
    process_ids = range(run_settings.num_chains)
    execute_mtmlda_on_procs = partial(execute_mtmlda_run,
                                      run_settings=run_settings,
                                      proposal_settings=settings.proposal_settings,
                                      accept_rate_settings=settings.accept_rate_settings,
                                      sampler_setup_settings=settings.sampler_setup_settings,
                                      sampler_run_settings=settings.sampler_run_settings,
                                      models=settings.models)
    with multiprocessing.Pool(processes=run_settings.num_chains) as process_pool:
        process_pool.map(execute_mtmlda_on_procs, process_ids)


if __name__ == "__main__":
    main()
