import time

import umbridge
from scipy.stats import multivariate_normal


class BananaPosterior(umbridge.Model):
    def __init__(self, model_name, sleep_time):
        self._sleep_time = sleep_time
        super().__init__(model_name)

    def get_input_sizes(self, config):
        return [2]

    def get_output_sizes(self, config):
        return [1]

    def supports_evaluate(self):
        return True

    def __call__(self, parameters, config={}):
        time.sleep(self._sleep_time)

        a = config.get("a", 2.0)
        b = config.get("b", 0.2)
        scale = config.get("scale", 1.0)
        y = [
            (parameters[0][0] / a),
            (parameters[0][1] * a + a * b * (parameters[0][0] ** 2 + a**2)),
        ]

        return [
            [
                multivariate_normal.logpdf(
                    y, [0, 4], [[1.0 * scale, 0.5 * scale], [0.5 * scale, 1.0 * scale]]
                )
            ]
        ]


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
