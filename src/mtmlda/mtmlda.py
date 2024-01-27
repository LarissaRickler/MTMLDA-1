from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import umbridge
from anytree import LevelOrderGroupIter, NodeMixin, util


sub_sampling_rates = [5, 3, -1]


class MLDASettings:
    rng_seed = 0
    num_levels = 3
    acceptance_rate_estimates = [0.2, 0.2, 0.2]


class RunSettings:
    num_samples = 5
    initial_state = np.array([1, 2, 3, 4])
        

class ThreadHandler:
    def __init__(self, worker_pool):
        pass


class MTMLDASampler:
    def __init__(self, mlda_settings, subsampling_generator, models, proposals):
        self._rng = np.random.default_rng(mlda_settings.rng_seed)
        self._num_levels = mlda_settings.num_levels
        self._subsampling_generator = subsampling_generator

    def run(self, run_settings):
        pass

    def _compute_accept_probability(
        self, new_state, current_state, proposal_new_current, proposal_current_new
    ):
        pass

    def _do_accept_reject(
        self, new_state, current_state, proposal_new_current, proposal_current_new
    ):
        pass
