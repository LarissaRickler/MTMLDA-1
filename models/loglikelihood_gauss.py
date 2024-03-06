import time
from typing import Any

import umbridge as ub


class GaussianLogLikelihood(ub.Model):
    def __init__(self) -> None:
        super().__init__("forward")
        self._mean = 0
        self._covariance = 1

    def get_input_sizes(self, config: Any) -> list[int]:
        return [1]

    def get_output_sizes(self, config: Any) -> list[int]:
        return [1]

    def supports_evaluate(self) -> bool:
        return True

    def __call__(self, parameters: list[list[float]], config: Any = {}) -> list[list[float]]:
        if config["meshFile"] == "model_0p1Hz":
            time.sleep(0.005)
        if config["meshFile"] == "model_0p3Hz":
            time.sleep(0.05)

        state_diff = parameters[0][0] - self._mean
        log_likelihood = -0.5 * state_diff**2 / self._covariance
        return [[log_likelihood]]


if __name__ == "__main__":
    ub.serve_models(
        [
            GaussianLogLikelihood(),
        ],
        port=4242,
        max_workers=100,
    )
