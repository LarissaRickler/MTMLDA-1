import argparse
import os
import time
from typing import Any

import umbridge as ub


# ==================================================================================================
def process_cli_arguments() -> bool:
    arg_parser = argparse.ArgumentParser(
        prog="loglikelihood_gauss.py",
        usage="python %(prog)s [options]",
        description="Umbridge server-side client emulating Gaussian log-likelihood",
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
        nargs=2,
        default=[0.001, 0.001],
        help="Sleep times to emulate simulation",
    )

    cli_args = arg_parser.parse_args()
    run_on_hq = cli_args.hyperqueue
    local_port = cli_args.port
    sleep_times = cli_args.sleep_times

    return run_on_hq, local_port, sleep_times


# ==================================================================================================
class GaussianLogLikelihood(ub.Model):
    def __init__(self, sleep_times: list[float]) -> None:
        super().__init__("gaussian_loglikelihood")
        self._time_coarse, self._time_fine = sleep_times
        self._mean = 5e6
        self._covariance = 1e12

    def get_input_sizes(self, _config: dict[str, Any] = {}) -> list[int]:
        return [1]

    def get_output_sizes(self, _config: dict[str, Any] = {}) -> list[int]:
        return [1]

    def supports_evaluate(self) -> bool:
        return True

    def __call__(
        self, parameters: list[list[float]], config: dict[str, Any] = {}
    ) -> list[list[float]]:
        if config["level"] == "0":
            time.sleep(self._time_coarse)
        if config["level"] == "1":
            time.sleep(self._time_fine)

        state_diff = parameters[0][0] - self._mean
        log_likelihood = -0.5 * state_diff**2 / self._covariance
        return [[float(log_likelihood)]]


# ==================================================================================================
def main():
    run_on_hq, local_port, sleep_times = process_cli_arguments()
    port = int(os.environ["PORT"]) if run_on_hq else local_port

    ub.serve_models(
        [
            GaussianLogLikelihood(sleep_times),
        ],
        port=port,
        max_workers=100,
    )


if __name__ == "__main__":
    main()
