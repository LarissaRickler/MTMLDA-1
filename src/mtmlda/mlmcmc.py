import numpy as np


# ==================================================================================================
class MLMetropolisHastingsKernel:
    def __init__(self, ground_proposal):
        self._ground_proposal = ground_proposal

    # ----------------------------------------------------------------------------------------------
    def compute_single_level_decision(self, node):
        new_state = node.state
        old_state = node.parent.state
        posterior_logp_new = node.logposterior
        posterior_logp_old = node.parent.logposterior
        proposal_logp_new_old = self._ground_proposal.evaluate_log_probability(new_state, old_state)
        proposal_logp_old_new = self._ground_proposal.evaluate_log_probability(old_state, new_state)

        accept_probability = min(
            1,
            np.exp(
                + posterior_logp_new
                + proposal_logp_old_new
                - posterior_logp_old
                - proposal_logp_new_old
            ),
        )
        accepted = node.parent.random_draw < accept_probability
        return accepted

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def compute_two_level_decision(node, same_level_parent):
        posterior_logp_new_fine = node.logposterior
        posterior_logp_old_coarse = same_level_parent.children[0].logposterior
        posterior_logp_old_fine = same_level_parent.logposterior
        posterior_logp_new_coarse = node.parent.logposterior

        accept_probability = min(
            1,
            np.exp(
                +posterior_logp_new_fine
                + posterior_logp_old_coarse
                - posterior_logp_old_fine
                - posterior_logp_new_coarse
            ),
        )
        accepted = node.parent.random_draw < accept_probability
        return accepted


# ==================================================================================================
class MLAcceptRateEstimator:
    def __init__(self, initial_guess, update_parameter):
        self._acceptance_rates = initial_guess
        self._update_parameter = update_parameter

    # ----------------------------------------------------------------------------------------------
    def get_acceptance_rate(self, node):
        acceptance_rate = self._acceptance_rates[node.level]
        return acceptance_rate

    # ----------------------------------------------------------------------------------------------
    def update(self, accepted, node):
        level = node.level
        decreased_rate = (1 - self._update_parameter) * self._acceptance_rates[level]
        if accepted:
            self._acceptance_rates[level] = decreased_rate + self._update_parameter
        else:
            self._acceptance_rates[level] = decreased_rate
