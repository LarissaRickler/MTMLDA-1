from dataclasses import dataclass
from pathlib import Path

import src.mtmlda.sampler as sampler


# ==================================================================================================
SamplerSetupSettings = sampler.SamplerSetupSettings
SamplerRunSettings = sampler.SamplerRunSettings

@dataclass
class ParallelRunSettings:
    num_chains: int
    result_directory_path: Path
    chain_file_stem: Path
    rng_state_save_file_stem: Path
    rng_state_load_file_stem: Path
    overwrite_results: bool