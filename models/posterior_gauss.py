import argparse
import os
import time
from typing import Any

import numpy as np
import umbridge as ub


# ==================================================================================================
def process_cli_arguments() -> bool:
    argParser = argparse.ArgumentParser(
        prog="posterior_gauss.py",
        usage="python %(prog)s [options]",
        description="Umbridge server-side client emulating Gaussian posterior",
    )

    argParser.add_argument(
        "-hq",
        "--hyperqueue",
        action="store_true",
        help="Run via Hyperqueue",
    )

    argParser.add_argument(
        "-p",
        "--port",
        type=float,
        required=False,
        default=4242,
        help="User-defined port (if not on Hyperqueue)",
    )

    argParser.add_argument(
        "-t",
        "--sleep_times",
        type=float,
        required=False,
        nargs=3,
        default=[0.03, 0.06, 1],
        help="Sleep times to emulate simulation",
    )

    cliArgs = argParser.parse_args()
    run_on_hq = cliArgs.hyperqueue
    local_port = cliArgs.port
    sleep_times = cliArgs.sleep_times

    return run_on_hq, local_port, sleep_times


# ==================================================================================================
class GaussianPosterior(ub.Model):
    def __init__(self, model_name: str, sleep_time: float) -> None:
        super().__init__(model_name)
        self._sleep_time = sleep_time
        self._mean = np.array([0, 0])
        self._precision = np.linalg.inv([[0.1, 0.05], [0.05, 0.1]])

    def get_input_sizes(self, config: Any) -> list[int]:
        return [2]

    def get_output_sizes(self, config: Any) -> list[int]:
        return [1]

    def supports_evaluate(self) -> bool:
        return True

    def __call__(self, parameters: list[list[float]], config: Any = {}) -> list[list[float]]:
        time.sleep(self._sleep_time)
        misfit = np.array(parameters[0]) - self._mean
        logp = -0.5 * misfit.T @ self._precision @ misfit
        return [[logp]]


# ==================================================================================================
def main():
    run_on_hq, local_port, sleep_times = process_cli_arguments()
    if run_on_hq:
        port = int(os.environ["PORT"])
    else:
        port = local_port

    ub.serve_models(
        [
            GaussianPosterior("gaussian_posterior_coarse", sleep_times[0]),
            GaussianPosterior("gaussian_posterior_intermediate", sleep_times[1]),
            GaussianPosterior("gaussian_posterior_fine", sleep_times[2]),
        ],
        port=port,
        max_workers=100,
    )


if __name__ == "__main__":
    main()
