import os
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pydot
import xarray as xa


# ==================================================================================================
postprocess_chain = True
chain_directory = Path("../results")
output_directory = Path("../results")
components = ["v1", "v2", "v3", "v4"]

dotfile_directory = Path("../results") / Path("mltree")
visualize_tree = False


# ==================================================================================================
def postprocess_chains(
    chain_directory: Path, output_directory: Path, components: list[str]
) -> None:
    dataset = _load_chain_data(chain_directory, components)
    _visualize_density_trace(dataset, output_directory, components)
    _visualize_data_pairs(dataset, output_directory)
    _visualize_autocorrelation(dataset, output_directory, components)
    _visualize_ess(dataset, output_directory, components)


def render_dot_files(dotfile_directory: Path) -> None:
    dot_files = _get_specific_file_type(dotfile_directory, "dot")
    dot_files = [dotfile_directory / Path(file) for file in dot_files]
    for file in dot_files:
        graph = pydot.graph_from_dot_file(file)[0]
        graph.write_png(file.with_suffix(".png"))


def _load_chain_data(chain_directory: Path, components: list[str]) -> xa.Dataset:
    npy_files = _get_specific_file_type(chain_directory, "npy")
    chains = np.array([np.load(chain_directory / Path(file)) for file in npy_files])
    datadict = {"mcmc_data": chains}
    dims = {"mcmc_data": ["components"]}
    coords = {"components": components}
    dataset = az.convert_to_dataset(datadict, dims=dims, coords=coords)
    return dataset


def _get_specific_file_type(directory: Path, file_type: str) -> list[str]:
    files = []
    for file in os.listdir(directory):
        if file.endswith(file_type):
            files.append(file)
    return files


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


def _visualize_autocorrelation(
    dataset: xa.Dataset, output_directory: Path, components: list[str]
) -> None:
    axes = az.plot_autocorr(dataset)
    figure = axes.flatten()[0].figure

    for i, component in enumerate(components):
        ax = axes.flatten()[i]
        ax.set_title(f"Autocorrelation for {component}")
        ax.set_xlabel("Sample number")
    figure.savefig(output_directory / Path("autocorrelation.pdf"))


def _visualize_ess(dataset: xa.Dataset, output_directory: Path, components: list[str]) -> None:
    num_draws = dataset.mcmc_data.shape[1]
    axes = az.plot_ess(dataset, kind="evolution", min_ess=num_draws)
    figure = axes.flatten()[0].figure

    for i, component in enumerate(components):
        ax = axes.flatten()[i]
        ax.set_title(f"ESS for {component}")
    figure.savefig(output_directory / Path("ess.pdf"))


def _visualize_data_pairs(
    dataset: xa.Dataset, output_directory: Path) -> None:
    axes = az.plot_pair(dataset, figsize=(15, 15))
    if isinstance(axes, np.ndarray):
        figure = axes[0, 0].figure
    else:
        figure = axes.figure
    figure.savefig(output_directory / Path("pair_plots.pdf"))


def main():
    if postprocess_chain:
        postprocess_chains(chain_directory, output_directory, components)
    if visualize_tree:
        render_dot_files(dotfile_directory)


if __name__ == "__main__":
    warnings.simplefilter('ignore', FutureWarning)
    main()
