import argparse
import os
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pydot
import xarray as xa
import matplotlib.pyplot as plt


# ==================================================================================================
def process_cli_arguments() -> tuple[str, str]:
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
    chain_directory = cliArgs.chain_directory
    tree_directory = cliArgs.tree_directory

    return chain_directory, tree_directory


# --------------------------------------------------------------------------------------------------
def postprocess_chains(chain_directory: Path) -> None:
    components, dataset = _load_chain_data(chain_directory)
    _visualize_density_trace(dataset, chain_directory, components)
    _visualize_autocorrelation(dataset, chain_directory, components)
    _visualize_ess(dataset, chain_directory, components)
    if len(components) > 1:
        _visualize_data_pairs(dataset, chain_directory)


# --------------------------------------------------------------------------------------------------
def render_dot_files(tree_directory: Path) -> None:
    dot_files = _get_specific_file_type(tree_directory, "dot")
    dot_files = [tree_directory / Path(file) for file in dot_files]
    for file in dot_files:
        graph = pydot.graph_from_dot_file(file)[0]
        graph.write_png(file.with_suffix(".png"))


# --------------------------------------------------------------------------------------------------
def _load_chain_data(chain_directory: Path) -> xa.Dataset:
    npy_files = _get_specific_file_type(chain_directory, "npy")
    chains = np.array([np.load(chain_directory / Path(file)) for file in npy_files])
    num_components = chains[0].shape[1]
    components = [f"component_{i}" for i in range(num_components)]

    datadict = {"mcmc_data": chains}
    dims = {"mcmc_data": ["components"]}
    coords = {"components": components}
    dataset = az.convert_to_dataset(datadict, dims=dims, coords=coords)

    return components, dataset


# --------------------------------------------------------------------------------------------------
def _get_specific_file_type(directory: Path, file_type: str) -> list[str]:
    files = []
    for file in os.listdir(directory):
        if file.endswith(file_type):
            files.append(file)
    return files


# --------------------------------------------------------------------------------------------------
def _visualize_density_trace(
    dataset: xa.Dataset, output_directory: Path, components: list[str]
) -> None:
    for component in components:
        axes = az.plot_trace(
            dataset, coords={"components": [component]}, show=False, figsize=(15, 5)
        )
        density_ax = axes.flatten()[0]
        trace_ax = axes.flatten()[1]
        figure = density_ax.figure
        density_ax.set_title(f"Posterior for {component}")
        density_ax.set_xlabel(component)
        trace_ax.set_title(f"MCMC trace for {component}")
        trace_ax.set_xlabel("Sample number")
        trace_ax.set_ylabel(component)
        figure.savefig(output_directory / Path(f"density_trace_{component}.pdf"))


# --------------------------------------------------------------------------------------------------
def _visualize_autocorrelation(
    dataset: xa.Dataset, output_directory: Path, components: list[str]
) -> None:
    axes = az.plot_autocorr(dataset)
    if len(components) == 1:
        axes = np.array((axes,))

    figure = axes.flatten()[0].figure
    figure.savefig(output_directory / Path("autocorrelation.pdf"))


# --------------------------------------------------------------------------------------------------
def _visualize_ess(dataset: xa.Dataset, output_directory: Path, components: list[str]) -> None:
    num_draws = dataset.mcmc_data.shape[1]
    axes = az.plot_ess(dataset, kind="evolution", min_ess=num_draws)
    if len(components) == 1:
        axes = np.array((axes,))

    figure = axes.flatten()[0].figure
    for i, component in enumerate(components):
        ax = axes.flatten()[i]
        ax.set_title(f"ESS for {component}")
    plt.tight_layout()
    figure.savefig(output_directory / Path("ess.pdf"))


# --------------------------------------------------------------------------------------------------
def _visualize_data_pairs(dataset: xa.Dataset, output_directory: Path) -> None:
    axes = az.plot_pair(dataset, figsize=(15, 15))
    if isinstance(axes, np.ndarray):
        figure = axes[0, 0].figure
    else:
        figure = axes.figure
    figure.savefig(output_directory / Path("pair_plots.pdf"))


# ==================================================================================================
def main():
    chain_directory, tree_directory = process_cli_arguments()

    if chain_directory is not None:
        postprocess_chains(Path(chain_directory))
    if tree_directory is not None:
        render_dot_files(Path(tree_directory))


if __name__ == "__main__":
    warnings.simplefilter("ignore", FutureWarning)
    main()
