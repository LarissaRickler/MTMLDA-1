"""Executable script for MTMLDA run postprocessing.

This script is an executable wrapper for the Postprocessor class. It performs postprocessing of
an MTMLDA run according to the provided settings files.
For info on how to run the script, type `python run.py --help` in the command line.

Functions:
    process_cli_arguments: Read in command-line arguments for application to run.
    main: Main routine to be invoked when script is executed
"""

import argparse
import warnings
from pathlib import Path

from src.mtmlda.run import postprocessor


# ==================================================================================================
def process_cli_arguments() -> str:
    """Read in command-line arguments for postprocessing settings, mainly data locations."""
    argParser = argparse.ArgumentParser(
        prog="postprocessing.py",
        usage="python %(prog)s [options]",
        description="Postprocessing file for parallel MLDA sampling",
    )

    argParser.add_argument(
        "-cdir",
        "--chain_directory",
        type=str,
        required=False,
        default=None,
        help="Directory containing the chains in npy format",
    )

    argParser.add_argument(
        "-tdir",
        "--tree_directory",
        type=str,
        required=False,
        default=None,
        help="Directory containing rendered trees in dot format",
    )

    argParser.add_argument(
        "-lag",
        "--acf_max_lag",
        type=int,
        required=False,
        default=100,
        help="Maximum lag for autocorrelation function",
    )

    cliArgs = argParser.parse_args()
    chain_directory = cliArgs.chain_directory
    tree_directory = cliArgs.tree_directory
    acf_max_lag = cliArgs.acf_max_lag
    if chain_directory is not None:
        chain_directory = Path(chain_directory)
    if tree_directory is not None:
        tree_directory = Path(tree_directory)

    return chain_directory, tree_directory, acf_max_lag


# ==================================================================================================
def main():
    """Entry point for the script, constructs and runs the Postprocessor."""
    chain_directory, tree_directory, acf_max_lag = process_cli_arguments()

    postprocessor_settings = postprocessor.PostprocessorSettings(
        chain_directory=chain_directory,
        tree_directory=tree_directory,
        output_data_directory=chain_directory,
        visualization_directory=chain_directory,
        acf_max_lag=acf_max_lag,
    )

    print("\n=== Start Postprocessing ===\n")
    pproc = postprocessor.Postprocessor(postprocessor_settings)
    pproc.run()
    print("\n============================\n")


if __name__ == "__main__":
    warnings.simplefilter("ignore", FutureWarning)
    main()
