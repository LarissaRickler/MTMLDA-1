from dataclasses import dataclass
from pathlib import Path
from itertools import combinations

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pydot
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from .. import utilities as utils

sns.set_theme(style="ticks")


# ==================================================================================================
@dataclass
class PostprocessorSettings:
    chain_directory: Path
    tree_directory: Path
    output_data_directory: Path
    visualization_directory: Path
    acf_max_lag: int
    ess_stride: int


# ==================================================================================================
class Postprocessor:
    def __init__(self, postprocessor_settings: PostprocessorSettings) -> None:
        self._chain_directory = postprocessor_settings.chain_directory
        self._tree_directory = postprocessor_settings.tree_directory
        self._output_data_directory = postprocessor_settings.output_data_directory
        self._visualization_directory = postprocessor_settings.visualization_directory
        self._acf_max_lag = postprocessor_settings.acf_max_lag
        self._ess_stride = postprocessor_settings.ess_stride

        (
            self._chains,
            self._num_components,
        ) = self._load_chain_data(postprocessor_settings.chain_directory)
        self._all_samples = np.concatenate(self._chains, axis=0)

    # ----------------------------------------------------------------------------------------------
    def run(self) -> None:
        if self._chain_directory is not None:
            print("Evaluate statistics ...")
            marginal_densities = self._compute_marginal_kdes()
            autocorrelations = self._compute_autocorrelation()
            effective_sample_size, true_sample_size = self._compute_effective_sample_size()

            if self._output_data_directory is not None:
                print("Save statistics data ...")
                self._save_data(
                    marginal_densities, autocorrelations, effective_sample_size, true_sample_size
                )
            if self._visualization_directory is not None:
                print("Generate plots ...")
                self._visualize_marginal_densities(marginal_densities)
                self._visualize_autocorrelation(autocorrelations)
                self._visualize_effective_sample_size(effective_sample_size, true_sample_size)
                if self._num_components > 1:
                    self._visualize_pairwise()

        if self._tree_directory is not None:
            print("Render dot files ...")
            self._render_dot_files(self._tree_directory)

    # ----------------------------------------------------------------------------------------------
    def _load_chain_data(self, chain_directory: Path) -> tuple[list[np.ndarray], int, int]:
        npy_files = utils.get_specific_file_type(chain_directory, "npy")
        chains = np.array([np.load(chain_directory / Path(file)) for file in npy_files])
        num_components = chains[0].shape[1]

        return chains, num_components

    # ----------------------------------------------------------------------------------------------
    def _compute_marginal_kdes(self) -> list:
        kdes = []
        for i in range(self._num_components):
            kdes.append(az.kde(self._all_samples[:, i], bw="scott"))

        return kdes

    # ----------------------------------------------------------------------------------------------
    def _compute_autocorrelation(self) -> list:
        autocorrelations = []
        for i in range(len(self._chains)):
            acf_per_chain = []
            for j in range(self._num_components):
                acf_per_chain.append(az.autocorr(self._chains[i][:, j]))
            autocorrelations.append(acf_per_chain)

        return autocorrelations

    # ----------------------------------------------------------------------------------------------
    def _compute_effective_sample_size(self) -> list:
        ess = []
        for i in range(self._num_components):
            ess_per_component = np.array(
                [
                    az.ess(self._all_samples[:n, i])
                    for n in range(4, len(self._all_samples), self._ess_stride)
                ]
            )
            ess.append(ess_per_component)

        true_sample_size = np.array(range(4, len(self._all_samples), self._ess_stride))
        return ess, true_sample_size

    # ----------------------------------------------------------------------------------------------
    def _save_data(
        self,
        marginal_densities: list,
        autocorrelations: list,
        effective_sample_size: list,
        true_sample_size: np.ndarray,
    ) -> None:
        marginal_densities_file = self._output_data_directory / Path("marginal_density.npz")
        autocorrelations_file = self._output_data_directory / Path("autocorrelation.npz")
        effective_sample_size_file = self._output_data_directory / Path("effective_sample_size.npz")

        md_data_dict = {}
        ac_data_dict = {}
        ess_data_dict = {}

        for i, density_per_component in enumerate(marginal_densities):
            md_data_dict[f"component_{i}"] = density_per_component
        for i , ess_per_component in enumerate(effective_sample_size):
            ess_data_dict[f"component_{i}"] = [true_sample_size, ess_per_component]
        for i, acf_per_chain in enumerate(autocorrelations):
            for j, acf_per_component in enumerate(acf_per_chain):
                ac_data_dict[f"chain_{i}_component_{j}"] = acf_per_component

        np.savez(marginal_densities_file, **md_data_dict)
        np.savez(autocorrelations_file, **ac_data_dict)
        np.savez(effective_sample_size_file, **ess_data_dict)
        
    # ----------------------------------------------------------------------------------------------
    def _visualize_marginal_densities(self, marginal_densities: list) -> None:
        visualization_file = self._visualization_directory / Path("marginal_density.pdf")
        with PdfPages(visualization_file) as pdf:
            for i, kde in enumerate(marginal_densities):
                fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
                ax.plot(kde[0], kde[1])
                ax.set_xlabel(rf"$\theta_{i+1}$")
                ax.set_ylabel(rf"$P_{{KDE}}(\theta_{i+1})$")
                pdf.savefig(fig)
                plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    def _visualize_autocorrelation(self, autocorrelations: list) -> None:
        visualization_file = self._visualization_directory / Path("autocorrelation.pdf")

        with PdfPages(visualization_file) as pdf:
            for i, acf in enumerate(autocorrelations):
                max_lag = min(self._acf_max_lag, len(acf[0]))
                lag_values = np.linspace(1, max_lag, max_lag)
                fig, axs = plt.subplots(
                    nrows=1, ncols=self._num_components, figsize=(8, 4), layout="constrained"
                )
                fig.suptitle(rf"Autocorrelation Chain {i}")
                for j, ax in enumerate(axs):
                    ax.bar(lag_values, acf[j][:max_lag])
                    ax.set_xlabel(r"Lag")
                    ax.set_ylabel(rf"Autocorrelation $\theta_{j+1}$")
                pdf.savefig(fig)
                plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    def _visualize_effective_sample_size(
        self, effective_sample_size: list, true_sample_size: np.ndarray
    ) -> None:
        visualization_file = self._visualization_directory / Path("effective_sample_size.pdf")

        with PdfPages(visualization_file) as pdf:
            for i, ess in enumerate(effective_sample_size):
                fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
                ax.plot(true_sample_size, ess)
                ax.set_xlabel(rf"Sample Size $\theta_{i+1}$")
                ax.set_ylabel(rf"Effective Sample Size $\theta_{i+1}$")
                pdf.savefig(fig)
                plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    def _visualize_pairwise(self):
        visualization_file = self._visualization_directory / Path("pairwise_samples.pdf")
        component_list = list(range(self._num_components))
        component_permutations = list(combinations(component_list, 2))

        with PdfPages(visualization_file) as pdf:
            for i, j in component_permutations:
                fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
                ax.scatter(self._all_samples[:, i], self._all_samples[:, j], s=10, alpha=0.1)
                ax.set_xlabel(rf"$\theta_{i+1}$")
                ax.set_ylabel(rf"$\theta_{j+1}$")
                pdf.savefig(fig)
                plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    def _render_dot_files(self, tree_directory: Path) -> None:
        dot_files = utils.get_specific_file_type(tree_directory, "dot")
        dot_files = [tree_directory / Path(file) for file in dot_files]
        for file in dot_files:
            graph = pydot.graph_from_dot_file(file)[0]
            graph.write_png(file.with_suffix(".png"))
