import umbridge
import numpy as np
from scipy.stats import multivariate_normal
import time
import os

# Inspired by https://github.com/chi-feng/mcmc-demo

class Banana(umbridge.Model):
    radius = 2.6
    sigma2 = 0.033

    def __init__(self):
        super().__init__("posterior")

    def get_input_sizes(self, config):
        return [2]

    def get_output_sizes(self, config):
        return [1]

    def __call__(self, parameters, config = {}):
        # Random sleep time multiplied by SLEEP_INTERVAL
        time.sleep(float(os.environ.get('SLEEP_INTERVAL', .1)) * np.random.random())

        a = config.get('a', 2.0)
        b = config.get('b', 0.2)
        scale = config.get('scale', 1.0)

        y = [(parameters[0][0] / a),
             (parameters[0][1] * a + a * b * (parameters[0][0]**2 + a**2))]

        return [[multivariate_normal.logpdf(y, [0, 4], [[1.0*scale, 0.5*scale], [0.5*scale, 1.0*scale]])]]

    def supports_evaluate(self):
        return True


class BananaCoarse(umbridge.Model):
    radius = 2.6
    sigma2 = 0.033

    def __init__(self):
        super().__init__("posterior_coarse")

    def get_input_sizes(self, config):
        return [2]

    def get_output_sizes(self, config):
        return [1]

    def __call__(self, parameters, config = {}):
        a = config.get('a', 2.0)
        b = config.get('b', 0.2)
        scale = config.get('scale', 1.5)

        y = [(parameters[0][0] / a),
             (parameters[0][1] * a + a * b * (parameters[0][0]**2 + a**2))]

        return [[multivariate_normal.logpdf(y, [0, 4], [[1.0*scale, 0.5*scale], [0.5*scale, 1.0*scale]])]]

    def supports_evaluate(self):
        return True

umbridge.serve_models([Banana(), BananaCoarse()], port=4243,max_workers=100)