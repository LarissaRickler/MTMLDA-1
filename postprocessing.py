"""Executable script for MTMLDA run postprocessing.

This script is an executable wrapper for the Postprocessor class. It performs postprocessing of
an MTMLDA run according to the provided settings files.
For info on how to run the script, type `python run.py --help` in the command line.

Functions:
    process_cli_arguments: Read in command-line arguments for application to run.
    main: Main routine to be invoked when script is executed
"""


import argparse
import importlib
import warnings

from src.mtmlda.run import postprocessor


# ==================================================================================================
def process_cli_arguments() -> str:
    """Read in command-line arguments for postprocessing settings."""
    argParser = argparse.ArgumentParser(
        prog="postprocessing.py",
        usage="python %(prog)s [options]",
        description="Postprocessing file for parallel MLDA sampling",
    )

    argParser.add_argument(
        "-dir",
        "--chain_directory",
        type=str,
        required=False,
        default=None,
        help="MCMC chain directory",
    )

    argParser.add_argument(
        "-tdir",
        "--tree_directory",
        type=str,
        required=False,
        default=None,
        help="Markov tree directory",
    )

    cliArgs = argParser.parse_args()
    application_dir = cliArgs.application.replace("/", ".").strip(".")
    settings_dir = f"{application_dir}.{cliArgs.settings}"

    return settings_dir


# ==================================================================================================
def main():
    """Entry point for the script, constructs and runs the Postprocessor."""
    settings_dir = process_cli_arguments()
    settings_module = importlib.import_module(settings_dir)

    print("\n=== Start Postprocessing ===\n")
    pproc = postprocessor.Postprocessor(settings_module.postprocessor_settings)
    pproc.run()
    print("\n============================\n")



if __name__ == "__main__":
    warnings.simplefilter("ignore", FutureWarning)
    main()
