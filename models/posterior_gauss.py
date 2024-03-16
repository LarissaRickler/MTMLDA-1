import argparse
import os
import time
from typing import Any

import umbridge as ub
import scipy.stats as stats


# ==================================================================================================
def process_cli_arguments() -> bool:
    argParser = argparse.ArgumentParser(
        prog="posterior_gauss.py",
        usage="python %(prog)s [options]",
        description="Umbridge server-side client emulating Gaussian posterior",
    )

    argParser.add_argument(
        "-c",
        "--cluster",
        action="store_true",
        help="Run via Hyperqueue",
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
    run_on_hq = cliArgs.cluster
    sleep_times = cliArgs.sleep_times
    return run_on_hq, sleep_times


# ==================================================================================================
class GaussianPosterior(ub.Model):
    def __init__(self, model_name: str, sleep_time: float) -> None:
        super().__init__(model_name)
        self._sleep_time = sleep_time
        mean = [0, 0]
        covariance = [[0.1, 0.05], [0.05, 0.1]]
        self._distribution = stats.multivariate_normal(mean, covariance)

    def get_input_sizes(self, config: Any) -> list[int]:
        return [2]

    def get_output_sizes(self, config: Any) -> list[int]:
        return [1]

    def supports_evaluate(self) -> bool:
        return True

    def __call__(self, parameters: list[list[float]], config: Any = {}) -> list[list[float]]:
        time.sleep(self._sleep_time)
        logp = self._distribution.logpdf(parameters[0])
        return [[logp]]


# ==================================================================================================
if __name__ == "__main__":
    run_on_hq, sleep_times = process_cli_arguments()
    if run_on_hq:
        port = int(os.environ["PORT"])
    else:
        port = 4242

    ub.serve_models(
        [
            GaussianPosterior("gaussian_posterior_coarse", sleep_times[0]),
            GaussianPosterior("gaussian_posterior_intermediate", sleep_times[1]),
            GaussianPosterior("gaussian_posterior_fine", sleep_times[2]),
        ],
        port=port,
        max_workers=100,
    )
