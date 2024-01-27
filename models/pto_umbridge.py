import umbridge
import numpy as np
import time


class BaseModel(umbridge.Model):
    _sleep_time = None

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


class FineModel(BaseModel):
    def __init__(self):
        self._sleep_time = 1
        super().__init__("parameter_to_observable_map_fine")


class IntermediateModel(BaseModel):
    def __init__(self):
        self._sleep_time = 0.6
        super().__init__("parameter_to_observable_map_intermediate")


class CoarseModel(BaseModel):
    def __init__(self):
        self._sleep_time = 0.3
        super().__init__("parameter_to_observable_map_coarse")


if __name__ == "__main__":
    umbridge.serve_models(
        [FineModel(), IntermediateModel(), CoarseModel()], port=4243, max_workers=100
    )
