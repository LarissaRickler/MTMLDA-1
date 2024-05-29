import os
import pickle
import time
from pathlib import Path
from typing import Any, Union

import numpy as np
import umbridge as ub


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
def load_pickle(process_id: int, load_path: Path):
    load_file = append_string_to_path(load_path, f"{process_id}.pkl")
    with load_file.open("rb") as load_file:
        object = pickle.load(load_file)
    return object


# --------------------------------------------------------------------------------------------------
def save_pickle(process_id: int, save_path: Path, object: Any, exist_ok: bool):
    os.makedirs(save_path.parent, exist_ok=exist_ok)
    save_file = append_string_to_path(save_path, f"{process_id}.pkl")
    with save_file.open("wb") as save_file:
        pickle.dump(object, save_file)


# --------------------------------------------------------------------------------------------------
def request_umbridge_server(process_id, address: str, name: str) -> ub.HTTPModel:
    server_available = False
    while not server_available:
        try:
            if process_id == 0:
                print(f"Calling server {name} at {address}...")
            ub_server = ub.HTTPModel(address, name)
            if process_id == 0:
                print("Server available\n")
            server_available = True
        except:
            time.sleep(10)

    return ub_server
