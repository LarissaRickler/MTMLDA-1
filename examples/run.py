"""Executable script for parallel MTMLDA runs.

This script is an executable wrapper for the ParallelRunner class. It executes chain-parallel
MTMLDA runs with Python's multiprocessing module.
For info on how to run the script, type `python run.py --help` in the command line.

Functions:
    process_cli_arguments: Read in command-line arguments for application to run.
    main: Main routine to be invoked when script is executed
"""

import argparse
import importlib

from mtmlda.run import runner


# ==================================================================================================
def process_cli_arguments() -> list[str]:
    """Read in command-line arguments for application to run.

    Every application has a builder and settings file to run. The user has to point to the directory
    where these files are stored. Per default, the run routine will search for the files
    `settings.py` and `builder.py` in the application directory. The user can provide different
    file names with the respective command line arguments.

    Returns:
        list[str]: strings for the directories of the settings and builder files
    """
    arg_parser = argparse.ArgumentParser(
        prog="run.py",
        usage="python %(prog)s [options]",
        description="Run file for parallel MLDA sampling",
    )

    arg_parser.add_argument(
        "-app",
        "--application",
        type=str,
        required=True,
        help="Application directory",
    )

    arg_parser.add_argument(
        "-s",
        "--settings",
        type=str,
        required=False,
        default="settings",
        help="Application settings file",
    )

    arg_parser.add_argument(
        "-b",
        "--builder",
        type=str,
        required=False,
        default="builder",
        help="Application builder file",
    )

    cli_args = arg_parser.parse_args()
    application_dir = cli_args.application.replace("/", ".").strip(".")

    dirs = []
    for module in (cli_args.settings, cli_args.builder):
        module_dir = f"{application_dir}.{module}"
        dirs.append(module_dir)

    return dirs


# ==================================================================================================
def main() -> None:
    """Main routine.

    The method reads in application files and runs chain-parallel MTMLDA runs within a
    multiprocessing pool. This functionality is implemented in the ParallelRunner class.
    """
    settings_dir, builder_dir = process_cli_arguments()
    settings_module = importlib.import_module(settings_dir)
    builder_module = importlib.import_module(builder_dir)

    print("\n====== Start Sampling ======\n")
    prunner = runner.ParallelRunner(
        application_builder=builder_module.ApplicationBuilder,
        parallel_run_settings=settings_module.parallel_run_settings,
        sampler_setup_settings=settings_module.sampler_setup_settings,
        sampler_run_settings=settings_module.sampler_run_settings,
        logger_settings=settings_module.logger_settings,
        inverse_problem_settings=settings_module.inverse_problem_settings,
        sampler_component_settings=settings_module.sampler_component_settings,
        initial_state_settings=settings_module.initial_state_settings,
    )
    prunner.run()
    print("\n============================\n")


if __name__ == "__main__":
    main()
