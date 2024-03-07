from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

import settings


# ==================================================================================================
class ApplicationBuilder:

    def __init__(self, process_id: int):
        self._process_id = process_id

    @abstractmethod
    def set_up_models(
        self, inverse_problem_settings: settings.InverseProblemSettings
    ) -> list[Callable]:
        pass

    @abstractmethod
    def set_up_sampler_components(
        self, sampler_component_settings: settings.SamplerComponentSettings
    ) -> tuple[Any, Any]:
        pass

    @abstractmethod
    def generate_initial_state(
        self, initial_state_settings: settings.InitialStateSettings
    ) -> np.ndarray:
        pass
