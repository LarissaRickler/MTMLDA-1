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
        "-c",
        "--cluster",
        action="store_true",
        help="Run via Hyperqueue",
    )

    cliArgs = argParser.parse_args()
    run_on_hq = cliArgs.cluster
    return run_on_hq


# ==================================================================================================
class PTOModel(ub.Model):
    _time_coarse = 0.005
    _time_fine = 0.05

    def __init__(self) -> None:
        super().__init__("forward")
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
            time.sleep(0.005)
        if config["meshFile"] == "model_0p3Hz":
            time.sleep(0.05)
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
if __name__ == "__main__":
    run_on_hq = process_cli_arguments()
    if run_on_hq:
        port = int(os.environ["PORT"])
    else:
        port = 4242

    ub.serve_models(
        [
            PTOModel(),
        ],
        port=port,
        max_workers=100,
    )
