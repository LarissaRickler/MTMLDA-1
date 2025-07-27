from dataclasses import dataclass
from pathlib import Path

from ..core import logging
from ..core import sampling


# ==================================================================================================
SamplerSetupSettings = sampling.SamplerSetupSettings
SamplerRunSettings = sampling.SamplerRunSettings
LoggerSettings = logging.LoggerSettings


@dataclass
class ParallelRunSettings:
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
