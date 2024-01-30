import time

import umbridge
from scipy.stats import multivariate_normal


class GaussianPosterior(umbridge.Model):
    def __init__(self, model_name, sleep_time):
        super().__init__(model_name)
        self._sleep_time = sleep_time
        self._mean = [0, 0]
        self._covariance = [[0.1, 0.05], [0.05, 0.1]]
        self._distribution = multivariate_normal(self._mean, self._covariance)

    def get_input_sizes(self, config):
        return [2]

    def get_output_sizes(self, config):
        return [1]

    def supports_evaluate(self):
        return True

    def __call__(self, parameters, config={}):
        time.sleep(self._sleep_time)
        logp = self._distribution.logpdf(parameters[0])
        return [[logp]]


if __name__ == "__main__":
    umbridge.serve_models(
        [
            GaussianPosterior(model_name="gauss_posterior_fine", sleep_time=0.1),
            GaussianPosterior(model_name="gauss_posterior_intermediate", sleep_time=0.06),
            GaussianPosterior(model_name="gauss_posterior_coarse", sleep_time=0.03),
        ],
        port=4243,
        max_workers=100,
    )
