"""Collection of all settings data classes.

Unified module for settings import from core library, apps, and run wrapper.
"""

from dataclasses import dataclass
from pathlib import Path

import src.mtmlda.logging as logging
import src.mtmlda.sampling as sampling

# ==================================================================================================
# Settings for components of the MTMLDA core library, see respective docstrings there for details.
SamplerSetupSettings = sampling.SamplerSetupSettings
SamplerRunSettings = sampling.SamplerRunSettings
LoggerSettings = logging.LoggerSettings


@dataclass
class ParallelRunSettings:
    """Data class for parallel run settings, used for the `run.py` wrapper.
    
    Attributes:
        num_chains (int): Number of parallel chains to run
        chain_save_path (Path): Path to save MCMC chain data, will be appended by process ID
        chain_load_path (Path): Path to load MCMC chain data, will be appended by process ID
        node_save_path (Path): Path to save node data, will be appended by process ID
        node_load_path (Path): Path to load node data, will be appended by process ID
        rng_state_save_path (Path): Path to save RNG state data, will be appended by process ID
        rng_state_load_path (Path): Path to load RNG state data, will be appended by process ID
        overwrite_chain (bool): Overwrite existing chain data
        overwrite_node (bool): Overwrite existing node data
        overwrite_rng_states (bool): Overwrite existing RNG state data
    """
    num_chains: int
    chain_save_path: Path
    chain_load_path: Path = None
    node_save_path: Path = None
    node_load_path: Path = None
    rng_state_save_path: Path = None
    rng_state_load_path: Path = None
    overwrite_chain: bool = True
    overwrite_node: bool = True
    overwrite_rng_states: bool = True
