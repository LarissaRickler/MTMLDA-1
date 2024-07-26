"""Executable script for parallel MTMLDA runs.

This script is an executable wrapper for combining applications with the MTMLDA sampling routine.
It allows for running parallel chains with Python's multiprocessing capabilities. All relevant
settings are adjusted according to the invoking process.
For info on how to run the script, type `python run.py --help` in the command line.

Functions:
    process_cli_arguments: Read in command-line arguments for application to run.
    execute_mtmlda_run: Main routine to execute MTMLDA runs.
    set_up_sampler: Set up MTMLDA sampler, with settings depending on invoking process.
    main: Main routine to be invoked when script is executed
"""

import argparse
import importlib
import multiprocessing
from collections.abc import Callable
from functools import partial
from typing import Any

import src.mtmlda.sampling as sampling
import utilities.utilities as utils
from components import abstract_builder, general_settings


# ==================================================================================================
def process_cli_arguments() -> list[str]:
    """Read in command-line arguments for application to run.

    Every application has a builder and settings file to run. The user has to point to the directory
    where these files are stored. Per default, the run routine will searach for the files 
    `settings.py` and `builder.py` in the application directory. The user can provide different 
    file names with the respective command line arguments.

    Returns:
        list[str]: strings of the directories of the settings and builder files
    """
    argParser = argparse.ArgumentParser(
        prog="run.py",
        usage="python %(prog)s [options]",
        description="Run file for parallel MLDA sampling",
    )

    argParser.add_argument(
        "-app",
        "--application",
        type=str,
        required=True,
        help="Application directory",
    )

    argParser.add_argument(
        "-s",
        "--settings",
        type=str,
        required=False,
        default="settings",
        help="Application settings file",
    )

    argParser.add_argument(
        "-b",
        "--builder",
        type=str,
        required=False,
        default="builder",
        help="Application builder file",
    )

    cliArgs = argParser.parse_args()
    application_dir = cliArgs.application.replace("/", ".").strip(".")

    dirs = []
    for module in (cliArgs.settings, cliArgs.builder):
        module_dir = f"{application_dir}.{module}"
        dirs.append(module_dir)

    return dirs


# --------------------------------------------------------------------------------------------------
def execute_mtmlda_run(
    process_id: int,
    application_builder: abstract_builder.ApplicationBuilder,
    parallel_run_settings: general_settings.ParallelRunSettings,
    sampler_setup_settings: general_settings.SamplerSetupSettings,
    sampler_run_settings: general_settings.SamplerRunSettings,
    logger_settings: general_settings.LoggerSettings,
    inverse_problem_settings: abstract_builder.InverseProblemSettings,
    sampler_component_settings: abstract_builder.SamplerComponentSettings,
    initial_state_settings: abstract_builder.InitialStateSettings,
) -> None:
    """Main routine to execute MTMLDA runs.

    The routine is called from a multiprocess pool and takes, next to generic settings, the id of
    the invoking process. All process-specific settings are adjusted according to that id.

    Args:
        process_id (int): id of the invoking process
        application_builder (abstract_builder.ApplicationBuilder): Application builder object
        parallel_run_settings (general_settings.ParallelRunSettings): Settings for parallel run
        sampler_setup_settings (general_settings.SamplerSetupSettings): Settings for setup of
            the MTMLDA sampler
        sampler_run_settings (general_settings.SamplerRunSettings): Settings for running the
            MTMLDA sampler
        logger_settings (general_settings.LoggerSettings): Settings for the MTMLDA logger
        inverse_problem_settings (abstract_builder.InverseProblemSettings): Settings for the setup
            of the posterior hierarchy of an application
        sampler_component_settings (abstract_builder.SamplerComponentSettings): Settings for setup
            of the MTMLDA sampler components specified by an application
        initial_state_settings (abstract_builder.InitialStateSettings): Settings for the generation
            of initial states for the Markov chains
    """
    # Set up posterior hierarchy
    app_builder = application_builder(process_id)
    models = app_builder.set_up_models(inverse_problem_settings)

    # Set up MTMLDA sampler
    ground_proposal, accept_rate_estimator = app_builder.set_up_sampler_components(
        sampler_component_settings
    )
    mtmlda_sampler = set_up_sampler(
        process_id,
        sampler_setup_settings,
        logger_settings,
        ground_proposal,
        accept_rate_estimator,
        models,
    )

    # Load structures for re-initialization
    if parallel_run_settings.rng_state_load_path is not None:
        rng_states = utils.load_pickle(process_id, parallel_run_settings.rng_state_load_path)
        mtmlda_sampler.set_rngs(rng_states)
    
    if parallel_run_settings.chain_load_path is not None:
        chain = utils.load_chain(process_id, parallel_run_settings.chain_load_path)
        initial_state = chain[-1, :]
    else:
        initial_state = app_builder.generate_initial_state(initial_state_settings)
    sampler_run_settings.initial_state = initial_state
    if parallel_run_settings.node_load_path is not None:
        initial_node = utils.load_pickle(process_id, parallel_run_settings.node_load_path)
        sampler_run_settings.initial_node = initial_node
    
    # Run sampler
    mcmc_chain, final_node = mtmlda_sampler.run(sampler_run_settings)
    rng_states = mtmlda_sampler.get_rngs()

    # Saver results and structures for re-initialization
    if parallel_run_settings.rng_state_save_path is not None:
        utils.save_pickle(
            process_id,
            parallel_run_settings.rng_state_save_path,
            rng_states,
            exist_ok=parallel_run_settings.overwrite_rng_states,
        )
    if parallel_run_settings.chain_save_path is not None:
        utils.save_chain(
            process_id,
            parallel_run_settings.chain_save_path,
            mcmc_chain,
            exist_ok=parallel_run_settings.overwrite_chain,
        )
    if parallel_run_settings.node_save_path is not None:
        utils.save_pickle(
            process_id,
            parallel_run_settings.node_save_path,
            final_node,
            exist_ok=parallel_run_settings.overwrite_node,
        )


# --------------------------------------------------------------------------------------------------
def set_up_sampler(
    process_id: int,
    sampler_setup_settings: general_settings.SamplerSetupSettings,
    logger_settings: general_settings.LoggerSettings,
    ground_proposal: Any,
    accept_rate_estimator: Any,
    models: list[Callable],
) -> sampling.MTMLDASampler:
    """Set up MTMLDA sampler, with settings depending on invoking process.

    Args:
        process_id (int): id of the invoking process
        sampler_setup_settings (general_settings.SamplerSetupSettings): Settings for MTMLDA sampler
        logger_settings (general_settings.LoggerSettings): Settings for MTMLDA logger
        ground_proposal (Any): Proposal object for coarse level chains
        accept_rate_estimator (Any): Multi-level accept rate estimator for prefetching
        models (list[Callable]): Posterior hierarchy as list of callables

    Returns:
        sampling.MTMLDASampler: Initialized MTMLDA sampler
    """
    # Adjust RNG seeds according to invoking process
    sampler_setup_settings.rng_seed_mltree = utils.distribute_rng_seeds_to_processes(
        sampler_setup_settings.rng_seed_mltree, process_id
    )
    sampler_setup_settings.rng_seed_node_init = utils.distribute_rng_seeds_to_processes(
        sampler_setup_settings.rng_seed_node_init, process_id
    )

    # Adjust file paths based on process id
    if logger_settings.logfile_path is not None:
        logger_settings.logfile_path = utils.append_string_to_path(
            logger_settings.logfile_path, f"chain_{process_id}.log"
        )
    if logger_settings.debugfile_path is not None:
        logger_settings.debugfile_path = utils.append_string_to_path(
            logger_settings.debugfile_path, f"chain_{process_id}.log"
        )
    if sampler_setup_settings.mltree_path is not None:
        sampler_setup_settings.mltree_path = utils.append_string_to_path(
            sampler_setup_settings.mltree_path, f"chain_{process_id}"
        )
    if process_id != 0:
        logger_settings.do_printing = False

    # Set up MTMLDA sampler
    mtmlda_sampler = sampling.MTMLDASampler(
        sampler_setup_settings,
        logger_settings,
        models,
        accept_rate_estimator,
        ground_proposal,
    )

    return mtmlda_sampler


# ==================================================================================================
def main() -> None:
    """Main routine.

    The method reads in application files and runs chain parallel MTMLDA runs with a multiprocessing
    pool.
    """
    settings_dir, builder_dir = process_cli_arguments()
    settings_module = importlib.import_module(settings_dir)
    builder_module = importlib.import_module(builder_dir)

    num_chains = settings_module.parallel_run_settings.num_chains
    process_ids = range(num_chains)

    print("\n=== Start Sampling ===\n")

    execute_mtmlda_on_procs = partial(
        execute_mtmlda_run,
        application_builder=builder_module.ApplicationBuilder,
        parallel_run_settings=settings_module.parallel_run_settings,
        sampler_setup_settings=settings_module.sampler_setup_settings,
        sampler_run_settings=settings_module.sampler_run_settings,
        logger_settings=settings_module.logger_settings,
        inverse_problem_settings=settings_module.inverse_problem_settings,
        sampler_component_settings=settings_module.sampler_component_settings,
        initial_state_settings=settings_module.initial_state_settings,
    )
    with multiprocessing.Pool(processes=num_chains) as process_pool:
        process_pool.map(execute_mtmlda_on_procs, process_ids)
    
    print("\n===== Finish Run =====\n")


if __name__ == "__main__":
    main()
