import argparse
import os
import time
from typing import Any

import numpy as np
import umbridge as ub


# ==================================================================================================
def process_cli_arguments() -> bool:
    argParser = argparse.ArgumentParser(
        prog="model_server.py",
        usage="python %(prog)s [options]",
        description="Umbridge server-side client emulating hierarchy of banana-shaped posteriors",
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
class BananaPosterior(ub.Model):
    def __init__(self, sleep_times: list[float]) -> None:
        super().__init__("banana_posterior")
        self._sleep_times = sleep_times
        self._mean = np.array([0, 0])
        self._precisions = [prefactor * np.identity(2) for prefactor in (0.1, 0.3, 1.0)]

    def get_input_sizes(self, config: dict[Any] = {}) -> list[int]:
        return [2]

    def get_output_sizes(self, config: dict[Any] = {}) -> list[int]:
        return [1]

    def supports_evaluate(self) -> bool:
        return True
    
    def _evaluate_logp(self, parameters: list[float], precision: np.ndarray) -> float:
        transformed_parameters = np.zeros((2,))
        transformed_parameters[0] = np.sqrt(20 * (parameters[0] ** 2 - 2 * parameters[1]) ** 2)
        transformed_parameters[1] = np.sqrt(2 * (parameters[1] - 0.25) ** 4)
        misfit = self._mean - transformed_parameters
        logp = -0.5 * misfit.T @ precision @ misfit
        return logp

    def __call__(self, parameters: list[list[float]], config: dict[Any] = {}) -> list[list[float]]:
        level = int(config["level"])
        time.sleep(self._sleep_times[level])
        logp = self._evaluate_logp(parameters[0], self._precisions[level])

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
            BananaPosterior(sleep_times=sleep_times),
        ],
        port=port,
        max_workers=100,
    )


if __name__ == "__main__":
    main()