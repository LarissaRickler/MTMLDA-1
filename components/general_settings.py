from dataclasses import dataclass
from pathlib import Path

import src.mtmlda.sampling as sampling

# ==================================================================================================
SamplerSetupSettings = sampling.SamplerSetupSettings
SamplerRunSettings = sampling.SamplerRunSettings


@dataclass(kw_only=True)
class ParallelRunSettings:
    num_chains: int
    result_directory_path: Path
    chain_file_stem: Path
    rng_state_save_file_stem: Path
    rng_state_load_file_stem: Path
    overwrite_results: bool
