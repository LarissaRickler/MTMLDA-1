import argparse
import importlib
import multiprocessing
from collections.abc import Callable
from functools import partial

from typing import Any

import utilities.utilities as utils
import src.mtmlda.sampling as sampling
from components import abstract_builder, general_settings


# ==================================================================================================
def process_cli_arguments() -> list[str]:
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
    app_builder = application_builder(process_id)
    models = app_builder.set_up_models(inverse_problem_settings)
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

    if parallel_run_settings.rng_state_load_path is not None:
        rng_states = utils.load_rng_states(process_id, parallel_run_settings.rng_state_load_path)
        mtmlda_sampler.set_rngs(rng_states)
    
    if parallel_run_settings.chain_load_path is not None:
        chain = utils.load_chain(process_id, parallel_run_settings.chain_load_path)
        initial_state = chain[-1, :]
    else:
        initial_state = app_builder.generate_initial_state(initial_state_settings)
    if parallel_run_settings.node_load_path is not None:
        initial_node = utils.load_node(process_id, parallel_run_settings.node_load_path)
        sampler_run_settings.initial_node = initial_node
    sampler_run_settings.initial_state = initial_state

    mcmc_chain, final_node = mtmlda_sampler.run(sampler_run_settings)
    rng_states = mtmlda_sampler.get_rngs()

    if parallel_run_settings.rng_state_save_path is not None:
        utils.save_rng_states(
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
        utils.save_node(
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
    sampler_setup_settings.rng_seed_mltree = utils.distribute_rng_seeds_to_processes(
        sampler_setup_settings.rng_seed_mltree, process_id
    )
    sampler_setup_settings.rng_seed_node_init = utils.distribute_rng_seeds_to_processes(
        sampler_setup_settings.rng_seed_node_init, process_id
    )

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
    print("\n======================\n")


if __name__ == "__main__":
    main()
