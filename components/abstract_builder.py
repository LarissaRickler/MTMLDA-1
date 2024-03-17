from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


# ==================================================================================================
@dataclass(kw_only=True)
class InverseProblemSettings:
    pass

@dataclass(kw_only=True)
class SamplerComponentSettings:
    pass

@dataclass(kw_only=True)
class InitialStateSettings:
    pass


# ==================================================================================================
class ApplicationBuilder:

    def __init__(self, process_id: int) -> None:
        self._process_id = process_id

    @abstractmethod
    def set_up_models(
        self, inverse_problem_settings: InverseProblemSettings
    ) -> list[Callable]:
        pass

    @abstractmethod
    def set_up_sampler_components(
        self, sampler_component_settings: SamplerComponentSettings
    ) -> tuple[Any, Any]:
        pass

    @abstractmethod
    def generate_initial_state(
        self, initial_state_settings: InitialStateSettings
    ) -> np.ndarray:
        pass
