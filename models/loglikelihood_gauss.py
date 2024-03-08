import argparse
import os
import time
from typing import Any

import umbridge as ub


# ==================================================================================================
def process_cli_arguments() -> bool:
    argParser = argparse.ArgumentParser(
        prog="loglikelihood_gauss.py",
        usage="python %(prog)s [options]",
        description="Umbridge server-side client emulating log-likelihood",
    )

    argParser.add_argument(
        "-c",
        "--cluster",
        action='store_true',
        help="Run via Hyperqueue",
    )

    cliArgs = argParser.parse_args()
    run_on_hq = cliArgs.cluster
    return run_on_hq


# ==================================================================================================
class GaussianLogLikelihood(ub.Model):
    def __init__(self) -> None:
        super().__init__("forward")
        self._mean = 5e6
        self._covariance = 1e12

    def get_input_sizes(self, config: Any) -> list[int]:
        return [1]

    def get_output_sizes(self, config: Any) -> list[int]:
        return [1]

    def supports_evaluate(self) -> bool:
        return True

    def __call__(self, parameters: list[list[float]], config: Any = {}) -> list[list[float]]:
        if config["order"] == 4:
            time.sleep(0.1)
        if config["order"] == 5:
            time.sleep(0.5)

        state_diff = parameters[0][0] - self._mean
        log_likelihood = -0.5 * state_diff**2 / self._covariance
        return [[log_likelihood]]


# ==================================================================================================
if __name__ == "__main__":
    run_on_hq = process_cli_arguments()
    if run_on_hq:
        port = int(os.environ["PORT"])
    else:
        port = 4242
    
    ub.serve_models(
        [
            GaussianLogLikelihood(),
        ],
        port=port,
        max_workers=100,
    )
