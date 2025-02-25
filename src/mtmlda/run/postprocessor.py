"""Postprocessing for parallel MTMLDA runs.

The postprocessing routines evaluate and visualize statistics and render trees for parallel MTMLDA
runs. All data can be saved for reproducibility of plots.

Classes:
    PostprocessorSettings: Dataclass to store postprocessing settings.
    Postprocessor: Postprocessor for parallel MTMLDA runs.
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pydot
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from mtmlda import utilities as utils

sns.set_theme(style="ticks")


# ==================================================================================================
@dataclass
class PostprocessorSettings:
    """Dataclass to store postprocessing settings.

    Attributes:
        chain_directory: Path
            Directory containing the chains in npy format. If `None`, no postprocessing is performed
        tree_directory: Path
            Directory containing exported Markov trees in dot format. If `None`, no rendering is
            performed.
        output_data_directory: Path
            Directory to save statistics data to. If `None`, no data is saved.
        visualization_directory: Path
            Directory to save visualizations to. If `None`, no visualizations are generated.
        acf_max_lag: int
            Maximum lag for autocorrelation function computation. Chains need to be longer than
            this value.
    """

    chain_directory: Path
    tree_directory: Path
    output_data_directory: Path
    visualization_directory: Path
    acf_max_lag: int


# ==================================================================================================
class Postprocessor:
    """Postprocessor for parallel MTMLDA runs.

    Given a set of directories, the postprocessor looks for chain data, evaluates statistics,
    visualizes these statistics, and renders potentially created Markov trees.

    Statistics:
    - Component-wise autocorrelation
    - Component-wise effective sample size
    - COmponent-wise potential scale reduction factor

    Visualizations:
    - 1D marginal densities
    - Pairwise sample distributions
    - ESS over sample size
    - PSRF over sample size
    - Component-wise ACFs
    - Markov trees
    """

    def __init__(self, postprocessor_settings: PostprocessorSettings) -> None:
        """Constructor of the Postprocessor.

        Reads in settings and loads chain data.

        Args:
            postprocessor_settings (PostprocessorSettings): Settings class for postprocessing
        """
        self._chain_directory = postprocessor_settings.chain_directory
        self._tree_directory = postprocessor_settings.tree_directory
        self._output_data_directory = postprocessor_settings.output_data_directory
        self._visualization_directory = postprocessor_settings.visualization_directory
        self._acf_max_lag = postprocessor_settings.acf_max_lag

        self._components = self._load_chain_data(postprocessor_settings.chain_directory)
        self._num_components = len(self._components)
        self._num_chains = self._components[0].shape[0]

    # ----------------------------------------------------------------------------------------------
    def run(self) -> None:
        """Main method of the postprocessor.

        Depending on which paths are provided, the postprocessor evaluates statistics, saves data,
        generates visualizations, and renders trees.
        """
        if self._chain_directory is not None:
            print("Evaluate statistics ...")
            marginal_densities = self._compute_marginal_kdes()
            autocorrelations = self._compute_autocorrelation()
            effective_sample_size, true_sample_size = self._compute_effective_sample_size()
            psrf, num_samples = self._compute_psrf()

            if self._output_data_directory is not None:
                print("Save statistics data ...")
                self._save_data(
                    marginal_densities,
                    autocorrelations,
                    effective_sample_size,
                    true_sample_size,
                    psrf,
                    num_samples,
                )
            if self._visualization_directory is not None:
                print("Generate plots ...")
                self._visualize_marginal_densities(marginal_densities)
                self._visualize_autocorrelation(autocorrelations)
                self._visualize_effective_sample_size(effective_sample_size, true_sample_size)
                self._visualize_psrf(psrf, num_samples)
                if self._num_components > 1:
                    self._visualize_pairwise()

        if self._tree_directory is not None:
            print("Render dot files ...")
            self._render_dot_files(self._tree_directory)

    # ----------------------------------------------------------------------------------------------
    def _load_chain_data(self, chain_directory: Path) -> list[np.ndarray]:
        """Loads chain data from specified directory.

        All `npy` files are interpreted as chains. Chains need to be 2D arrays, where the first
        dimension corresponds to the number of samples and the second dimension to the number of
        components. Chains of different length are NOT allowed.

        Args:
            chain_directory (Path): Directory of raw chain data.

        Raises:
            FileNotFoundError: Checks that all chains have correct format.

        Returns:
            list[np.ndarray]: Samples ordered by component.
        """
        npy_files = utils.get_specific_file_type(chain_directory, "npy")
        try:
            chains = [np.load(chain_directory / Path(file)) for file in npy_files]
        except FileNotFoundError:
            raise FileNotFoundError("Chain files couldn't be loaded.") from None
        assert all(len(chain.shape) == 2 for chain in chains), "Chains need to be 2D arrays."
        num_components = chains[0].shape[1]
        assert min([chain.shape[0] for chain in chains]) >= self._acf_max_lag, (
            "Chains need to be longer than the maximum lag for autocorrelation."
        )

        components = []
        for component in range(num_components):
            component_samples = np.stack([chain[:, component] for chain in chains], axis=0)
            components.append(component_samples)

        return components

    # ----------------------------------------------------------------------------------------------
    def _compute_marginal_kdes(self) -> list:
        """Compute 1D marginal densities through KDE.

        Note that the samples from all chains are concatenated, including those that might be
        discarded as burn-in.

        Returns:
            list: 1D marginal densities for each component.
        """
        kdes = []
        for component_samples in self._components:
            component_kde = az.kde(component_samples.flatten(), bw="scott")
            kdes.append(component_kde)

        return kdes

    # ----------------------------------------------------------------------------------------------
    def _compute_autocorrelation(self) -> list:
        """Computes ACFs for eacht chain and component.

        Useful for assessing mixing/burn-in of chains.

        Returns:
            list: List of lists containing ACFs for each component and chain.
        """
        autocorrelations = []
        for component_samples in self._components:
            acf_per_component = [
                az.autocorr(component_samples[i, :]) for i in range(self._num_chains)
            ]
            autocorrelations.append(acf_per_component)

        return autocorrelations

    # ----------------------------------------------------------------------------------------------
    def _compute_effective_sample_size(self) -> list:
        """Evaluates the ESS per component.

        Evaluation is started at a sample size of 4 and is increased in steps of 1% of the total
        sample size per chain.

        Returns:
            list: ESS for all components.
        """
        num_samples_per_chain = self._components[0].shape[1]
        num_chains = self._components[0].shape[0]
        stride = int(np.ceil(num_samples_per_chain / 100))

        effective_sample_size = []
        for component in self._components:
            ess_per_component = np.array(
                [az.ess(component[:, :n]) for n in range(4, num_samples_per_chain, stride)]
            )
            effective_sample_size.append(ess_per_component)

        true_sample_size = np.arange(4, num_samples_per_chain * num_chains, stride * num_chains)
        return effective_sample_size, true_sample_size

    # ----------------------------------------------------------------------------------------------
    def _compute_psrf(self) -> list:
        """Computes the potential scale reduction factor (PSRF) for each component.

        Evaluation is started at a sample size of 4 and is increased in steps of 1% of the total
        sample size per chain.
        """
        num_samples_per_chain = self._components[0].shape[1]
        stride = int(np.ceil(num_samples_per_chain / 100))

        psrf = []
        for component in self._components:
            rhat_per_component = np.array(
                [az.rhat(component[:, :n]) for n in range(4, num_samples_per_chain, stride)]
            )
            psrf.append(rhat_per_component)

        num_samples = np.arange(4, num_samples_per_chain, stride)
        return psrf, num_samples

    # ----------------------------------------------------------------------------------------------
    def _save_data(
        self,
        marginal_densities: list,
        autocorrelations: list,
        effective_sample_size: list,
        true_sample_size: np.ndarray,
        psrf: list,
        num_samples: np.ndarray,
    ) -> None:
        """Save statistics data for reproducibility of plots.

        Args:
            marginal_densities (list): 1D Marginal densities from KDE.
            autocorrelations (list): ACFs for all components.
            effective_sample_size (list): ESS for all components.
            true_sample_size (np.ndarray): True sample size array for comparison to ESS.
            psrf (list): PSRF for all components.
            num_samples (np.ndarray): Sample size array for comparison to PSRF.
        """
        marginal_densities_file = self._output_data_directory / Path("marginal_density.npz")
        autocorrelations_file = self._output_data_directory / Path("autocorrelation.npz")
        effective_sample_size_file = self._output_data_directory / Path("effective_sample_size.npz")
        psrf_file = self._output_data_directory / Path("psrf.npz")

        md_data_dict = {}
        ac_data_dict = {}
        ess_data_dict = {}
        psrf_data_dict = {}

        for i, density_per_component in enumerate(marginal_densities):
            md_data_dict[f"component_{i}"] = density_per_component
        for i, ess_per_component in enumerate(effective_sample_size):
            ess_data_dict[f"component_{i}"] = [true_sample_size, ess_per_component]
        for i, psrf_per_component in enumerate(psrf):
            psrf_data_dict[f"component_{i}"] = [num_samples, psrf_per_component]
        for i, acf_per_component in enumerate(autocorrelations):
            for j, acf_per_chain in enumerate(acf_per_component):
                ac_data_dict[f"component_{i}_chain_{j}"] = acf_per_chain

        np.savez(marginal_densities_file, **md_data_dict)
        np.savez(autocorrelations_file, **ac_data_dict)
        np.savez(effective_sample_size_file, **ess_data_dict)
        np.savez(psrf_file, **psrf_data_dict)

    # ----------------------------------------------------------------------------------------------
    def _visualize_marginal_densities(self, marginal_densities: list) -> None:
        """Visualize 1D Marginals for each component.

        Args:
            marginal_densities (list): 1D densities, estimated via KDE.
        """
        visualization_file = self._visualization_directory / Path("marginal_density.pdf")
        with PdfPages(visualization_file) as pdf:
            for i, kde in enumerate(marginal_densities):
                fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
                ax.plot(kde[0], kde[1])
                ax.set_xlabel(rf"$\theta_{i + 1}$")
                ax.set_ylabel(rf"$P_{{KDE}}(\theta_{i + 1})$")
                pdf.savefig(fig)
                plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    def _visualize_autocorrelation(self, autocorrelations: list) -> None:
        """Visualize ACFs for all components.

        Args:
            autocorrelations (list): Computed ACFs for all components.
        """
        visualization_file = self._visualization_directory / Path("autocorrelation.pdf")
        num_chains = len(autocorrelations[0])

        with PdfPages(visualization_file) as pdf:
            for i, acf in enumerate(autocorrelations):
                max_lag = min(self._acf_max_lag, acf[0].size)
                lag_values = np.linspace(1, max_lag, max_lag)
                fig, axs = plt.subplots(
                    nrows=1,
                    ncols=num_chains,
                    figsize=(4 * num_chains, 4),
                    layout="constrained",
                )
                if num_chains == 1:
                    axs = [
                        axs,
                    ]
                fig.suptitle(rf"Autocorrelation $\theta_{i + 1}$")
                for j, ax in enumerate(axs):
                    ax.bar(lag_values, acf[j][:max_lag])
                    ax.set_xlabel(r"Lag")
                    ax.set_ylabel(rf"ACF chain {j}")
                pdf.savefig(fig)
                plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    def _visualize_effective_sample_size(
        self, effective_sample_size: list, true_sample_size: np.ndarray
    ) -> None:
        """Visualize ESS for every component.

        Args:
            effective_sample_size (list): ESS for all components.
            true_sample_size (np.ndarray): True sample size for comparison.
        """
        visualization_file = self._visualization_directory / Path("effective_sample_size.pdf")

        with PdfPages(visualization_file) as pdf:
            for i, ess in enumerate(effective_sample_size):
                fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
                ax.plot(true_sample_size, ess)
                ax.set_xlabel(r"$N$")
                ax.set_ylabel(rf"ESS $\theta_{i + 1}$")
                pdf.savefig(fig)
                plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    def _visualize_psrf(self, psrf: list, num_samples: np.ndarray) -> None:
        """Visualize PSRF for every component.

        Args:
            psrf (list): PSRF for all components.
            num_samples (np.ndarray): Number of samples over which to plot.
        """
        visualization_file = self._visualization_directory / Path("psrf.pdf")

        with PdfPages(visualization_file) as pdf:
            for i, component_psrf in enumerate(psrf):
                fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
                ax.plot(num_samples, component_psrf)
                ax.set_xlabel(r"$N$")
                ax.set_ylabel(rf"$\hat{{R}}(\theta_{i + 1})$")
                pdf.savefig(fig)
                plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    def _visualize_pairwise(self) -> None:
        """Visualize all perwise sample distribution in an all-vs-all fashion."""
        visualization_file = self._visualization_directory / Path("pairwise_samples.pdf")
        component_list = list(range(self._num_components))
        component_permutations = list(combinations(component_list, 2))

        with PdfPages(visualization_file) as pdf:
            for i, j in component_permutations:
                fig, ax = plt.subplots(figsize=(4, 4), layout="constrained")
                ax.scatter(
                    self._components[i].flatten(), self._components[j].flatten(), s=10, alpha=0.1
                )
                ax.set_xlabel(rf"$\theta_{i + 1}$")
                ax.set_ylabel(rf"$\theta_{j + 1}$")
                pdf.savefig(fig)
                plt.close(fig)

    # ----------------------------------------------------------------------------------------------
    def _render_dot_files(self, tree_directory: Path) -> None:
        """Render tree files with Pydot.

        Note: Rendering trees is quite time-consuming.

        Args:
            tree_directory (Path): Path to stored dot files. Rendered tree images are stored in the
                same directory.
        """
        dot_files = utils.get_specific_file_type(tree_directory, "dot")
        dot_files = [tree_directory / Path(file) for file in dot_files]
        for file in dot_files:
            graph = pydot.graph_from_dot_file(file)[0]
            graph.write_png(file.with_suffix(".png"))
