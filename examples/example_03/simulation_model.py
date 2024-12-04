import argparse
import os
import time
from typing import Any

import numpy as np
import umbridge as ub


# ==================================================================================================
def process_cli_arguments() -> bool:
    arg_parser = argparse.ArgumentParser(
        prog="model_server.py",
        usage="python %(prog)s [options]",
        description="Umbridge server-side client emulating hierarchy of banana-shaped posteriors",
    )

    arg_parser.add_argument(
        "-hq",
        "--hyperqueue",
        action="store_true",
        help="Run via Hyperqueue",
    )

    arg_parser.add_argument(
        "-p",
        "--port",
        type=float,
        required=False,
        default=4242,
        help="User-defined port (if not on Hyperqueue)",
    )

    arg_parser.add_argument(
        "-t",
        "--sleep_times",
        type=float,
        required=False,
        nargs=3,
        default=[0.001, 0.001, 0.001],
        help="Sleep times to emulate simulation",
    )

    cli_args = arg_parser.parse_args()
    run_on_hq = cli_args.hyperqueue
    local_port = cli_args.port
    sleep_times = cli_args.sleep_times

    return run_on_hq, local_port, sleep_times


# ==================================================================================================
class BananaPosterior(ub.Model):
    def __init__(self, sleep_times: list[float]) -> None:
        super().__init__("banana_posterior")
        self._sleep_times = sleep_times
        self._mean = np.array([0, 0])
        self._precisions = [prefactor * np.identity(2) for prefactor in (0.1, 0.3, 1.0)]

    def get_input_sizes(self, _config: dict[Any] = {}) -> list[int]:
        return [2]

    def get_output_sizes(self, _config: dict[Any] = {}) -> list[int]:
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

        return [[float(logp)]]


# ==================================================================================================
def main():
    run_on_hq, local_port, sleep_times = process_cli_arguments()
    port = int(os.environ["PORT"]) if run_on_hq else local_port

    ub.serve_models(
        [
            BananaPosterior(sleep_times=sleep_times),
        ],
        port=port,
        max_workers=100,
    )


if __name__ == "__main__":
    main()