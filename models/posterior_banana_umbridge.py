import time
from typing import Any

import umbridge
from scipy.stats import multivariate_normal


class BananaPosterior(umbridge.Model):
    def __init__(self, model_name: str, sleep_time: float) -> None:
        self._sleep_time = sleep_time
        super().__init__(model_name)
        mean = [0, 4]
        covariance = [[1, 0.5], [0.5, 1]]
        self._distribution = multivariate_normal(mean, covariance)

    def get_input_sizes(self, config: Any = {}) -> list[int]:
        return [2]

    def get_output_sizes(self, config: Any = {}) -> list[int]:
        return [1]

    def supports_evaluate(self) -> bool:
        return True

    def __call__(self, parameters: list[list[float]], config: Any = {}) -> list[list[float]]:
        time.sleep(self._sleep_time)
        transformed_input = [
            (parameters[0][0] / 2),
            (parameters[0][1] * 2 + 0.4 * (parameters[0][0] ** 2 + 4)),
        ]
        logp = self._distribution.logpdf(transformed_input)

        return [[logp]]


if __name__ == "__main__":
    umbridge.serve_models(
        [
            BananaPosterior(model_name="banana_posterior_fine", sleep_time=0.1),
            BananaPosterior(model_name="banana_posterior_intermediate", sleep_time=0.06),
            BananaPosterior(model_name="banana_posterior_coarse", sleep_time=0.03),
        ],
        port=4243,
        max_workers=100,
    )
