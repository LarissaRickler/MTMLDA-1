import time
from typing import Any

import numpy as np
import umbridge as ub


class PTOModel(ub.Model):
    def __init__(self) -> None:
        super().__init__("forward")

    def get_input_sizes(self, config: Any = {}) -> list[int]:
        return [4]

    def get_output_sizes(self, config: Any = {}) -> list[int]:
        return [100]

    def supports_evaluate(self) -> bool:
        return True

    def __call__(self, parameters: list[list[float]], config: str) -> list[list[float]]:
        if config == "coarse":
            time.sleep(0.3)
        if config == "fine":
            time.sleep(0.3)
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
            PTOModel(),
            PTOModel(),
        ],
        port=4243,
        max_workers=100,
    )
