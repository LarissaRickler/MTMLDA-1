import os
import pickle
from pathlib import Path
from typing import Union

import numpy as np


# ==================================================================================================
def distribute_rng_seeds_to_processes(seeds: Union[list, int], process_id: int):
    if isinstance(seeds, list):
        return seeds[process_id]
    else:
        return int(seeds * process_id)


# --------------------------------------------------------------------------------------------------
def append_string_to_path(path: Path, string: int) -> Path:
    extended_path = path.with_name(f"{path.name}_{string}")
    return extended_path


# --------------------------------------------------------------------------------------------------
def load_chain(process_id: int, load_path: Path) -> None:
    chain_file = append_string_to_path(load_path, f"{process_id}.npy")
    chain = np.load(chain_file)
    return chain


# --------------------------------------------------------------------------------------------------
def save_chain(
    process_id: int, save_path: Path, mcmc_trace: list[np.ndarray], exist_ok: bool
) -> None:
    os.makedirs(save_path.parent, exist_ok=exist_ok)
    chain_file = append_string_to_path(save_path, f"{process_id}.npy")
    np.save(chain_file, mcmc_trace)


# --------------------------------------------------------------------------------------------------
def load_rng_states(process_id, load_path):
    rng_state_file = append_string_to_path(load_path, f"{process_id}.pkl")
    with rng_state_file.open("rb") as rng_state_file:
        rng_states = pickle.load(rng_state_file)
    return rng_states


# --------------------------------------------------------------------------------------------------
def save_rng_states(process_id, save_path, rng_states, exist_ok):
    os.makedirs(save_path.parent, exist_ok=exist_ok)
    rng_state_file = append_string_to_path(save_path, f"{process_id}.pkl")
    with rng_state_file.open("wb") as rng_state_file:
        pickle.dump(rng_states, rng_state_file)
