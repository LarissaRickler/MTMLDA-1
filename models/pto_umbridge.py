import umbridge
import numpy as np
import time


class PTOModel(umbridge.Model):
    def __init__(self, model_name, sleep_time):
        self._sleep_time = sleep_time
        super().__init__(model_name)

    def get_input_sizes(self, config):
        return [4]

    def get_output_sizes(self, config):
        return [100]

    def supports_evaluate(self):
        return True

    def __call__(self, parameters, config={}):
        time.sleep(self._sleep_time)
        observables = self._transform_input_to_output(parameters)
        return [observables.tolist()]

    def _transform_input_to_output(self, parameters):
        observables = np.zeros((100,))
        const_block = np.ones((25,))
        observables[0:25] = parameters[0][0] * const_block
        observables[25:50] = parameters[0][1] * const_block
        observables[50:75] = parameters[0][2] * const_block
        observables[75:100] = parameters[0][3] * const_block

        return observables


if __name__ == "__main__":
    umbridge.serve_models(
        [
            PTOModel(model_name="parameter_to_observable_map_fine", sleep_time=1),
            PTOModel(model_name="parameter_to_observable_map_intermediate", sleep_time=0.6),
            PTOModel(model_name="parameter_to_observable_map_coarse", sleep_time=0.3),
        ],
        port=4243,
        max_workers=100,
    )
