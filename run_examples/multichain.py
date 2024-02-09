import multiprocessing
import os
import pickle
import sys
from functools import partial
from pathlib import Path

sys.path.append(str(Path("../").resolve()))

import numpy as np

import src.mtmlda.mcmc as mcmc
import src.mtmlda.sampler as sampler
import settings


# ==================================================================================================
class run_settings:
    num_chains = 4
    result_directory_path = Path("results")
    chain_file_stem = Path("chain")
    rng_state_save_file_stem = Path("rng_states")
    rng_state_load_file_stem = Path("rng_states")
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
        process_id,
        run_settings,
        proposal_settings,
        accept_rate_settings,
        sampler_setup_settings,
        models,
    )
    mcmc_chain = mtmlda_sampler.run(sampler_run_settings)
    _save_results(process_id, run_settings, mcmc_chain, mtmlda_sampler)


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


def _set_up_sampler(
    process_id,
    run_settings,
    proposal_settings,
    accept_rate_settings,
    sampler_setup_settings,
    models,
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
    if run_settings.rng_state_load_file_stem is not None:
        rng_state_file = (
            run_settings.result_directory_path
            / run_settings.rng_state_load_file_stem.with_name(
                f"{run_settings.rng_state_load_file_stem.name}_{process_id}.pkl"
            )
        )
        with rng_state_file.open("rb") as rng_state_file:
            rng_states = pickle.load(rng_state_file)
        mtmlda_sampler.set_rngs(rng_states)
    return mtmlda_sampler


def _save_results(process_id, run_settings, mcmc_trace, mtmlda_sampler) -> None:
    os.makedirs(run_settings.result_directory_path, exist_ok=run_settings.overwrite_results)
    chain_file = run_settings.result_directory_path / run_settings.chain_file_stem.with_name(
        f"{run_settings.chain_file_stem.name}_{process_id}.npy"
    )
    np.save(chain_file, mcmc_trace)

    if run_settings.rng_state_save_file_stem is not None:
        rng_state_file = (
            run_settings.result_directory_path
            / run_settings.rng_state_save_file_stem.with_name(
                f"{run_settings.rng_state_save_file_stem.name}_{process_id}.pkl"
            )
        )
        with rng_state_file.open("wb") as rng_state_file:
            pickle.dump(mtmlda_sampler.get_rngs(), rng_state_file)


# --------------------------------------------------------------------------------------------------
def main() -> None:
    process_ids = range(run_settings.num_chains)
    execute_mtmlda_on_procs = partial(
        execute_mtmlda_run,
        run_settings=run_settings,
        proposal_settings=settings.proposal_settings,
        accept_rate_settings=settings.accept_rate_settings,
        sampler_setup_settings=settings.sampler_setup_settings,
        sampler_run_settings=settings.sampler_run_settings,
        models=settings.models,
    )
    with multiprocessing.Pool(processes=run_settings.num_chains) as process_pool:
        process_pool.map(execute_mtmlda_on_procs, process_ids)


if __name__ == "__main__":
    main()
