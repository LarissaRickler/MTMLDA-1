import time

import numpy as np
import umbridge as ub


class PTOModel(ub.Model):
    _time_coarse = 0.005
    _time_fine = 0.05

    def __init__(self) -> None:
        super().__init__("forward")
        self._parameter_ranges = [[500, 2000], [1, 20], [20e9, 30e9], [20e9, 30e9]]

    def get_input_sizes(self, config):
        return [4]

    def get_output_sizes(self, config):
        return [4]

    def supports_evaluate(self) -> bool:
        return True

    def __call__(self, parameters, config):
        if config["meshFile"] == "model_0p1Hz":
            time.sleep(0.005)
        if config["meshFile"] == "model_0p3Hz":
            time.sleep(0.05)
        scaled_parameters = self._scale_input(parameters)
        observables = self._compute_observables(scaled_parameters)
        return [observables]

    def _scale_input(self, parameters):
        for i, value in enumerate(parameters[0]):
            lower_bound, upper_bound = self._parameter_ranges[i]
            parameters[0][i] = 4 * (value - lower_bound) / (upper_bound - lower_bound) - 2
        return parameters

    def _compute_observables(self, parameters: list[list[float]]) -> np.ndarray:
        parameters = parameters[0]
        observables = [0,] * len(parameters)
        
        observables[0] = np.sqrt(20*(parameters[0]**2 - 2*parameters[1])**2)
        observables[1] = np.sqrt(2*(parameters[1] - 0.25)**4)
        observables[2] = np.sqrt(20*(parameters[3]**2 - 2*parameters[2])**2)
        observables[3] = np.sqrt(2*(parameters[2] - 0.25)**4)
        return observables


if __name__ == "__main__":
    ub.serve_models(
        [
            PTOModel(),
        ],
        port=4242,
        max_workers=100,
    )
