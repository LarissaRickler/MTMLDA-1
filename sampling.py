import argparse
import importlib
import multiprocessing
import os
import pickle
from functools import partial

import numpy as np

import src.mtmlda.sampler as sampler
from components import builder, settings


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
def execute_mtmlda_run(
    process_id,
    application_builder,
    parallel_run_settings,
    inverse_problem_settings,
    sampler_component_settings,
    sampler_setup_settings,
    sampler_run_settings,
):

    app_builder = application_builder(process_id)
    models = app_builder.set_up_models(inverse_problem_settings)
    initial_state = app_builder.generate_initial_state()
    ground_proposal, accept_rate_estimator = app_builder.set_up_sampler_components(
        sampler_component_settings
    )

    mtmlda_sampler = set_up_sampler(
        process_id,
        parallel_run_settings,
        sampler_setup_settings,
        ground_proposal,
        accept_rate_estimator,
        models,
    )

    sampler_run_settings.rng_seed_node_init = process_id
    sampler_run_settings.initial_state = initial_state
    mcmc_chain = mtmlda_sampler.run(sampler_run_settings)
    save_results(process_id, parallel_run_settings, mcmc_chain, mtmlda_sampler)


# --------------------------------------------------------------------------------------------------
def set_up_sampler(
    process_id,
    parallel_run_settings,
    sampler_setup_settings,
    ground_proposal,
    accept_rate_estimator,
    models,
):
    sampler_setup_settings.rng_seed_mltree = process_id
    if process_id != 0:
        sampler_setup_settings.logfile_path = None
        sampler_setup_settings.mltree_path = None
        sampler_setup_settings.do_printing = False

    mtmlda_sampler = sampler.MTMLDASampler(
        sampler_setup_settings,
        models,
        accept_rate_estimator,
        ground_proposal,
    )

    if parallel_run_settings.rng_state_load_file_stem is not None:
        rng_state_file = (
            parallel_run_settings.result_directory_path
            / parallel_run_settings.rng_state_load_file_stem.with_name(
                f"{parallel_run_settings.rng_state_load_file_stem.name}_{process_id}.pkl"
            )
        )
        with rng_state_file.open("rb") as rng_state_file:
            rng_states = pickle.load(rng_state_file)
        mtmlda_sampler.set_rngs(rng_states)
    return mtmlda_sampler


# --------------------------------------------------------------------------------------------------
def save_results(process_id, parallel_run_settings, mcmc_trace, mtmlda_sampler):
    os.makedirs(
        parallel_run_settings.result_directory_path,
        exist_ok=parallel_run_settings.overwrite_results,
    )
    chain_file = (
        parallel_run_settings.result_directory_path
        / parallel_run_settings.chain_file_stem.with_name(
            f"{parallel_run_settings.chain_file_stem.name}_{process_id}.npy"
        )
    )
    np.save(chain_file, mcmc_trace)

    if parallel_run_settings.rng_state_save_file_stem is not None:
        rng_state_file = (
            parallel_run_settings.result_directory_path
            / parallel_run_settings.rng_state_save_file_stem.with_name(
                f"{parallel_run_settings.rng_state_save_file_stem.name}_{process_id}.pkl"
            )
        )
        with rng_state_file.open("wb") as rng_state_file:
            pickle.dump(mtmlda_sampler.get_rngs(), rng_state_file)


# ==================================================================================================
def main():
    application = process_cli_arguments()
    settings_module = importlib.import_module(f"{application}.settings")
    builder_module = importlib.import_module(f"{application}.builder")

    num_chains = settings_module.parallel_run_settings.num_chains
    process_ids = range(num_chains)
    execute_mtmlda_on_procs = partial(
        execute_mtmlda_run,
        application_builder=builder_module.ApplicationBuilder,
        parallel_run_settings=settings_module.parallel_run_settings,
        inverse_problem_settings=settings_module.inverse_problem_settings,
        sampler_component_settings=settings_module.sampler_component_settings,
        sampler_setup_settings=settings_module.sampler_setup_settings,
        sampler_run_settings=settings_module.sampler_run_settings,
    )
    with multiprocessing.Pool(processes=num_chains) as process_pool:
        process_pool.map(execute_mtmlda_on_procs, process_ids)


if __name__ == "__main__":
    main()
