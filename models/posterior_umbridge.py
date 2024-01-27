import time

import umbridge
from scipy.stats import multivariate_normal


class BananaBase(umbridge.Model):
    _sleep_time = None

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


class BananaFine(umbridge.Model):
    def __init__(self):
        self._sleep_time = 1
        super().__init__("posterior_fine")


class BananaIntermediate(umbridge.Model):
    def __init__(self):
        self._sleep_time = 0.6
        super().__init__("posterior_intermediate")


class BananaCoarse(umbridge.Model):
    def __init__(self):
        self._sleep_time = 0.3
        super().__init__("posterior_coarse")


if __name__ == "__main__":
    umbridge.serve_models(
        [BananaFine(), BananaIntermediate(), BananaCoarse()], port=4243, max_workers=100
    )
