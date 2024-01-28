class MLMetropolisHastingsKernel:
    @staticmethod
    def compute_single_level_decision(node):
        pass

    @staticmethod
    def compute_two_level_decision(node, same_level_parent):
        pass


class MLAcceptRateEstimator:
    def __init__(self, initial_guess, update_parameter):
        self._acceptance_rates = initial_guess
        self._update_parameter = update_parameter

    def get_acceptance_rate(self, node):
        acceptance_rate = self._acceptance_rates[node.level]
        return acceptance_rate

    def update(self, accepted, level):
        decreased_rate = (1 - self._update_parameter) * self._acceptance_rates[level]
        if accepted:
            self._acceptance_rates[level] = decreased_rate + self._update_parameter
        else:
            self._acceptance_rates[level] = decreased_rate
