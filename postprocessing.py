import argparse
import importlib
import warnings

from src.mtmlda.run import postprocessor


# ==================================================================================================
def process_cli_arguments() -> str:
    argParser = argparse.ArgumentParser(
        prog="postprocessing.py",
        usage="python %(prog)s [options]",
        description="Postprocessing file for parallel MLDA sampling",
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

    cliArgs = argParser.parse_args()
    application_dir = cliArgs.application.replace("/", ".").strip(".")
    module_dir = f"{application_dir}.{cliArgs.settings}"

    return module_dir


# ==================================================================================================
def main():
    settings_dir = process_cli_arguments()
    settings_module = importlib.import_module(settings_dir)

    print("\n=== Start Postprocessing ===\n")
    pproc = postprocessor.Postprocessor(settings_module.postprocessor_settings)
    pproc.run()
    print("\n============================\n")


if __name__ == "__main__":
    warnings.simplefilter("ignore", FutureWarning)
    main()
