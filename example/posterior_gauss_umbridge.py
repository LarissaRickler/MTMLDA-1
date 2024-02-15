import time
from typing import Any

import umbridge as ub
import scipy.stats as stats


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


if __name__ == "__main__":
    ub.serve_models(
        [
            GaussianPosterior(model_name="gauss_posterior_fine", sleep_time=0.1),
            GaussianPosterior(model_name="gauss_posterior_intermediate", sleep_time=0.06),
            GaussianPosterior(model_name="gauss_posterior_coarse", sleep_time=0.03),
        ],
        port=4243,
        max_workers=100,
    )
