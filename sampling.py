import argparse
import importlib
import multiprocessing
import os
import pickle
from functools import partial

import numpy as np

import src.mtmlda.sampler as sampler


# ==================================================================================================
def process_cli_arguments():
    argParser = argparse.ArgumentParser(
        prog="sampling.py",
        usage="python %(prog)s [options]",
        description="Run file for parallel MLDA sampling",
    )

    argParser.add_argument(
        "-app",
        "--application",
        type=str,
        required=True,
        help="Application directory in dot notation",
    )

    cliArgs = argParser.parse_args()
    application = cliArgs.application
    return application


# --------------------------------------------------------------------------------------------------
def execute_mtmlda_run(process_id, settings, component_setup):
    if process_id == 0:
        print("Modify process-dependent settings")
    modify_process_dependent_settings(
        process_id,
        settings.proposal_settings,
        settings.sampler_setup_settings,
        settings.sampler_run_settings,
        settings.prior_settings,
    )

    if process_id == 0:
        print("Set up models")
    models, prior = component_setup.set_up_models(
        process_id, settings.model_settings, settings.prior_settings, settings.likelihood_settings
    )
    settings.sampler_run_settings.initial_state = prior.sample()

    if process_id == 0:
        print("Set up sampler")
    ground_proposal, accept_rate_estimator = component_setup.set_up_sampler_components(
        settings.proposal_settings, settings.accept_rate_settings
    )
    mtmlda_sampler = set_up_sampler(
        process_id,
        settings.run_settings,
        settings.sampler_setup_settings,
        ground_proposal,
        accept_rate_estimator,
        models,
    )

    if process_id == 0:
        print("Start sampling")
    mcmc_chain = mtmlda_sampler.run(settings.sampler_run_settings)

    if process_id == 0:
        print("Save results")
    save_results(process_id, settings.run_settings, mcmc_chain, mtmlda_sampler)


# --------------------------------------------------------------------------------------------------
def modify_process_dependent_settings(
    process_id, proposal_settings, sampler_setup_settings, sampler_run_settings, prior_settings
):
    proposal_settings.rng_seed = process_id
    sampler_setup_settings.rng_seed_mltree = process_id
    sampler_run_settings.rng_seed_node_init = process_id
    prior_settings.rng_seed = process_id

    if process_id != 0:
        sampler_setup_settings.logfile_path = None
        sampler_setup_settings.mltree_path = None
        sampler_setup_settings.do_printing = False


# --------------------------------------------------------------------------------------------------
def set_up_sampler(
    process_id,
    run_settings,
    sampler_setup_settings,
    ground_proposal,
    accept_rate_estimator,
    models,
):
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


# --------------------------------------------------------------------------------------------------
def save_results(process_id, run_settings, mcmc_trace, mtmlda_sampler):
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


# ==================================================================================================
def main():
    application = process_cli_arguments()
    settings_module = importlib.import_module(f"{application}.settings")
    setup_module = importlib.import_module(f"{application}.component_setup")
    settings = settings_module.Settings
    component_setup = setup_module.ComponentSetup

    num_chains = settings.run_settings.num_chains
    process_ids = range(num_chains)
    execute_mtmlda_on_procs = partial(
        execute_mtmlda_run, settings=settings, component_setup=component_setup
    )
    with multiprocessing.Pool(processes=num_chains) as process_pool:
        process_pool.map(execute_mtmlda_on_procs, process_ids)


if __name__ == "__main__":
    main()
