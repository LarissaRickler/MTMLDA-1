import functools
import multiprocessing
import os
import pathlib

import numpy as np

# --------------------------------------------------------------------------------------------------
num_chains = 2


class Settings:
    num_parameters = 4
    parameter_ranges = np.array([[500, 5000], [1e5, 1e6], [5, 50], [5, 50]])
    output_directory = "results"
    output_filename = "chain"
    overwrite_results = True


# --------------------------------------------------------------------------------------------------
def execute_mtmlda_run(process_id, settings):
    initial_state = _get_initial_state(process_id, settings)
    mcmc_trace = _run_chain(process_id, initial_state, settings)
    _save_trace(process_id, mcmc_trace, settings)


def _get_initial_state(process_id, settings):
    rng = np.random.default_rng(process_id)
    initial_state = [
        rng.uniform(low=settings.parameter_ranges[i, 0], high=settings.parameter_ranges[i, 1])
        for i in range(settings.num_parameters)
    ]
    return initial_state


def _run_chain(process_id, initial_state, settings):
    return process_id * initial_state


def _save_trace(process_id, mcmc_trace, settings):
    output_directory = pathlib.Path(settings.output_directory) / pathlib.Path(
        f"{settings.output_filename}_{process_id}"
    )
    os.makedirs(output_directory.parent, exist_ok=settings.overwrite_results)
    np.save(output_directory, mcmc_trace)


# --------------------------------------------------------------------------------------------------
def main():
    process_ids = range(num_chains)
    run_per_process = functools.partial(execute_mtmlda_run, settings=Settings)
    with multiprocessing.Pool(processes=num_chains) as process_pool:
        process_pool.map(run_per_process, process_ids)


if __name__ == "__main__":
    main()
