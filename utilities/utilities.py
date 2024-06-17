"""_summary_."""

import os
import pickle
import time
from numbers import Real
from pathlib import Path
from typing import Any, Union

import numpy as np
import umbridge as ub


# ==================================================================================================
def distribute_rng_seeds_to_processes(seeds: Union[list, int], process_id: int):
    """Distribute and optionall modify seeds for different parallel processes.

    This is a generic function that distributes RNG seeds to parallel processes when running MCMC
    with multiple independent chains. It can be used in two different ways. If the seeds are
    specified as a list, each process will receive the seed at the corresponding index. If the seeds
    are specified as a single value, each process will receive a seed that is the product of that
    value and the process ID.

    Args:
        seeds (Union[list, int]): RNG seeds to distribute to processes, list or single value
        process_id (int): Integer ID of the calling process

    Returns:
        Real: RNG seed for the calling process 
    """
    assert isinstance(process_id, int) and process_id >= 0, "Process ID must be a positive integer"
    if isinstance(seeds, list):
        assert all(
            isinstance(seed, Real) for seed in seeds
        ), "All seeds must be real numbers when specified in list"
        return seeds[process_id]
    else:
        assert isinstance(seeds, Real), "Seed must be real number when specified as single value"
        return int(seeds * process_id)


# --------------------------------------------------------------------------------------------------
def append_string_to_path(path: Path, string: int) -> Path:
    """Extend a Path object with a file name extension provided as a string.
    
    The new path object's name is extended by "_string". This routine is used to create process-
    specific file names by appending the ID of the calling process. Such a convention is also
    assumed when loading files.

    Args:
        path (Path): Path object
        string (int): String extending the file name

    Returns:
        Path: New path, name extended by "_string"
    """
    extended_path = path.with_name(f"{path.name}_{string}")
    assert isinstance(extended_path, Path), "Extended path must be a pathlib.Path object"
    return extended_path


# --------------------------------------------------------------------------------------------------
def load_chain(process_id: int, load_path: Path) -> None:
    """Load MCMC chain data from npy file.

    The provided file path is extended by the process ID via the `append_string_to_path` function.
    Clearly, this method only works if it is called by the exact same number of chains that have
    been saved before.

    Args:
        process_id (int): ID of the calling process
        load_path (Path): Generic path to find chain files in

    Raises:
        FileNotFoundError: Checks if the chain file exists

    Returns:
        np.ndarray: MCMC chain
    """
    chain_file = append_string_to_path(load_path, f"{process_id}.npy")
    try:
        chain = np.load(chain_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Chain file not found for process id {process_id}")
    return chain


# --------------------------------------------------------------------------------------------------
def save_chain(
    process_id: int, save_path: Path, mcmc_trace: list[np.ndarray], exist_ok: bool
) -> None:
    """Save MCMC chain arry to npy file.

    The save path is process-specific through the `append_string_to_path` function.

    Args:
        process_id (int): ID of the calling process
        save_path (Path): Generic path to save chain files to
        mcmc_trace (list[np.ndarray]): MCMC chain array to save
        exist_ok (bool): Choose if existing file should be overwritten
    """
    os.makedirs(save_path.parent, exist_ok=exist_ok)
    chain_file = append_string_to_path(save_path, f"{process_id}.npy")
    np.save(chain_file, mcmc_trace)


# --------------------------------------------------------------------------------------------------
def load_pickle(process_id: int, load_path: Path) -> Any:
    """Load a pickled object into memory.

    The provided file path is extended by the process ID via the `append_string_to_path` function.
    This method only works if it is called by the exact same number of processes that have
    been used for saving. The method is generic, and used for loading rng states and Markov tree
    nodes.

    Args:
        process_id (int): ID of the calling process
        load_path (Path): Generic load path

    Raises:
        FileNotFoundError: Checks if the specified file exists

    Returns:
        Any: Loaded object
    """
    load_file = append_string_to_path(load_path, f"{process_id}.pkl")
    try:
        with load_file.open("rb") as load_file:
            object = pickle.load(load_file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found for process id {process_id}")
    return object


# --------------------------------------------------------------------------------------------------
def save_pickle(process_id: int, save_path: Path, object: Any, exist_ok: bool) -> None:
    """Save a generic object into pickle.

    The save path is process-specific through the `append_string_to_path` function. The method is
    generic, and used for saving rng states and Markov tree nodes.

    Args:
        process_id (int): ID of the calling process
        save_path (Path): Generic save path
        object (Any): Object to save
        exist_ok (bool): Choose if existing object should be overwritten
    """
    os.makedirs(save_path.parent, exist_ok=exist_ok)
    save_file = append_string_to_path(save_path, f"{process_id}.pkl")
    with save_file.open("wb") as save_file:
        pickle.dump(object, save_file)


# --------------------------------------------------------------------------------------------------
def request_umbridge_server(process_id: int, address: str, name: str) -> ub.HTTPModel:
    """Request umbrdige model server with fail-save for long response times.

    The function tries to connect to an umbridge server at the provided address and name. It is a
    simple wrapper to the 'HTTPModel' class constructor. It repeatedly tries to connect to the
    server until it is available.

    Args:
        process_id (int): ID of the calling process
        address (str): Address of the umbridge server to call
        name (str): Name of the umbridge server to call

    Returns:
        ub.HTTPModel: Umbridge server object
    """
    server_available = False
    while not server_available:
        try:
            if process_id == 0:
                print(f"Calling server {name} at {address}...")
            ub_server = ub.HTTPModel(address, name)
            if process_id == 0:
                print("Server available\n")
            server_available = True
        except Exception:
            time.sleep(10)

    return ub_server
