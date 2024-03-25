import argparse
import os
import time
from typing import Any

import numpy as np
import umbridge as ub


# ==================================================================================================
def process_cli_arguments() -> bool:
    argParser = argparse.ArgumentParser(
        prog="pto_banana.py",
        usage="python %(prog)s [options]",
        description="Umbridge server-side client emulating PTO map",
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
        nargs=2,
        default=[0.005, 0.05],
        help="Sleep times to emulate simulation",
    )

    cliArgs = argParser.parse_args()
    run_on_hq = cliArgs.hyperqueue
    local_port = cliArgs.port
    sleep_times = cliArgs.sleep_times

    return run_on_hq, local_port, sleep_times


# ==================================================================================================
class PTOModel(ub.Model):

    def __init__(self, sleep_times: list[float]) -> None:
        super().__init__("forward")
        self._time_coarse, self._time_fine = sleep_times
        self._parameter_ranges = [[500, 2000], [1, 20], [20e9, 30e9], [20e9, 30e9]]

    def get_input_sizes(self, config: dict[str:Any] = {}) -> list[float]:
        return [4]

    def get_output_sizes(self, config: dict[str:Any] = {}) -> list[float]:
        return [4]

    def supports_evaluate(self) -> bool:
        return True

    def __call__(
        self, parameters: list[list[float]], config: dict[str:Any] = {}
    ) -> list[list[float]]:
        if config["meshFile"] == "model_0p1Hz":
            time.sleep(self._time_coarse)
        if config["meshFile"] == "model_0p3Hz":
            time.sleep(self._time_fine)
        scaled_parameters = self._scale_input(parameters)
        observables = self._compute_observables(scaled_parameters)
        return observables

    def _scale_input(self, parameters: list[list[float]]) -> list[list[float]]:
        for i, value in enumerate(parameters[0]):
            lower_bound, upper_bound = self._parameter_ranges[i]
            parameters[0][i] = 4 * (value - lower_bound) / (upper_bound - lower_bound) - 2
        return parameters

    def _compute_observables(self, parameters: list[list[float]]) -> list[list[float]]:
        parameters = parameters[0]
        observables = [
            0,
        ] * len(parameters)

        observables[0] = np.sqrt(20 * (parameters[0] ** 2 - 2 * parameters[1]) ** 2)
        observables[1] = np.sqrt(2 * (parameters[1] - 0.25) ** 4)
        observables[2] = np.sqrt(20 * (parameters[3] ** 2 - 2 * parameters[2]) ** 2)
        observables[3] = np.sqrt(2 * (parameters[2] - 0.25) ** 4)
        return [observables]


# ==================================================================================================
def main():
    run_on_hq, local_port, sleep_times = process_cli_arguments()
    if run_on_hq:
        port = int(os.environ["PORT"])
    else:
        port = local_port

    ub.serve_models(
        [
            PTOModel(sleep_times),
        ],
        port=port,
        max_workers=100,
    )


if __name__ == "__main__":
    main()
