import time
from typing import Any

import numpy as np
import umbridge as ub


class PTOModel(ub.Model):
    def __init__(self, model_name: str, sleep_time: float) -> None:
        self._sleep_time = sleep_time
        super().__init__(model_name)

    def get_input_sizes(self, config: Any = {}) -> list[int]:
        return [4]

    def get_output_sizes(self, config: Any = {}) -> list[int]:
        return [100]

    def supports_evaluate(self) -> bool:
        return True

    def __call__(self, parameters: list[list[float]], config: Any = {}) -> list[list[float]]:
        time.sleep(self._sleep_time)
        observables = self._transform_input_to_output(parameters)
        return [observables.tolist()]

    def _transform_input_to_output(self, parameters: list[list[float]]) -> np.ndarray:
        observables = np.zeros((100,))
        const_block = 0.1 * np.ones((25,))
        observables[0:25] = parameters[0][0] * const_block
        observables[25:50] = parameters[0][1] * const_block
        observables[50:75] = parameters[0][2] * const_block
        observables[75:100] = parameters[0][3] * const_block

        return observables


if __name__ == "__main__":
    ub.serve_models(
        [
            PTOModel(model_name="parameter_to_observable_map_fine", sleep_time=1),
            PTOModel(model_name="parameter_to_observable_map_coarse", sleep_time=0.3),
        ],
        port=4243,
        max_workers=100,
    )
