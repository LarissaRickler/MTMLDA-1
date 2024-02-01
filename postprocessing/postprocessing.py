import os
from pathlib import Path

import arviz as az
import numpy as np
import pydot


# ==================================================================================================
chain_directory = Path("../results")
output_directory = Path(".")
components = ["x", "y"]

dotfile_directory = Path("../results") / Path("mltree")
visualize_tree = False


# ==================================================================================================
def postprocess_chains(chain_directory, output_directory, components):
    dataset = _load_chain_data(chain_directory, components)
    _visualize_density_trace(dataset, output_directory, components)
    _visualize_autocorrelation(dataset, output_directory, components)
    _visualize_ess(dataset, output_directory, components)

    if dataset.mcmc_data.shape[2] == 2:
        _visualize_data_pairs(dataset, output_directory, components)

def render_dot_files(dotfile_directory):
    dot_files = _get_specific_file_type(dotfile_directory, "dot")
    for file in dot_files:
        graph = pydot.graph_from_dot_file(file)[0]
        graph.write_png(file.with_suffix(".png"))

def _load_chain_data(chain_directory, components):
    npy_files = _get_specific_file_type(chain_directory, "npy")
    chains = np.array([np.load(chain_directory / Path(file)) for file in npy_files])
    datadict = {"mcmc_data": chains}
    dims = {"mcmc_data": ["components"]}
    coords = {"components": components}
    dataset = az.convert_to_dataset(datadict, dims=dims, coords=coords)
    return dataset

def _get_specific_file_type(directory, file_type):
    files = []
    for file in os.listdir(directory):
        if file.endswith(file_type):
            files.append(file)
    return files

def _visualize_density_trace(dataset, output_directory, components):
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

def _visualize_autocorrelation(dataset, output_directory, components):
    axes = az.plot_autocorr(dataset)
    figure = axes.flatten()[0].figure

    for i, component in enumerate(components):
        ax = axes.flatten()[i]
        ax.set_title(f"Autocorrelation for {component}")
        ax.set_xlabel("Sample number")
    figure.savefig(output_directory/ Path("autocorrelation.pdf"))

def _visualize_ess(dataset, output_directory, components):
    num_draws = dataset.mcmc_data.shape[1]
    axes = az.plot_ess(dataset, kind="evolution", min_ess=num_draws)
    figure = axes.flatten()[0].figure

    for i, component in enumerate(components):
        ax = axes.flatten()[i]
        ax.set_title(f"ESS for {component}")
    figure.savefig(output_directory / Path("ess.pdf"))

def _visualize_data_pairs(dataset, output_directory, components):
    ax = az.plot_pair(dataset, figsize=(7, 7))
    figure = ax.figure
    ax.set_xlabel(components[0])
    ax.set_ylabel(components[1])
    figure.savefig(output_directory / Path("2D_data_pairs.pdf"))
    

def main():
    postprocess_chains(chain_directory, output_directory, components)
    if visualize_tree:
        render_dot_files(dotfile_directory)


if __name__ == "__main__":
    main()
