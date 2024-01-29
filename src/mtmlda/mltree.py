import numpy as np
from anytree import LevelOrderGroupIter, NodeMixin, util


class MTNodeBase:
    probability_reached = None
    state = None
    logposterior = None
    logposterior_coarse = None
    computing = False
    random_draw = None
    level = -1
    subchain_index = 0


class MTNode(MTNodeBase, NodeMixin):
    def __init__(self, name, parent=None, children=None):
        super(MTNodeBase, self).__init__()
        self.name = name
        self.parent = parent
        if children:
            self.children = children


class MLTreeSearchFunctions:
    def __init__(self, num_levels):
        self._num_levels = num_levels

    @staticmethod
    def find_max_probability_node(root):
        max_probability = 0.0
        max_node = None
        for node in root.leaves:
            if node.name == "a":
                parent_probability_reached = 1.0
                if node.parent is not None:
                    parent_probability_reached = node.parent.probability_reached
                if (
                    parent_probability_reached > max_probability
                    and node.logposterior is None
                    and not node.computing
                ):
                    max_probability = parent_probability_reached
                    max_node = node
        return max_node

    @staticmethod
    def get_same_level_parent(node):
        if node.parent is None:
            return None
        same_level_parent = node.parent
        while same_level_parent.level != node.level:
            if same_level_parent.parent is None:
                return None
            same_level_parent = same_level_parent.parent
        return same_level_parent

    @staticmethod
    def get_same_level_parent_child_on_path(node):
        same_level_parent = node.parent
        child = node
        while same_level_parent.level != node.level:
            child = same_level_parent
            same_level_parent = same_level_parent.parent
        return child

    @staticmethod
    def get_unique_fine_level_child(node, num_levels):
        if len(node.children) != 1:
            return None
        unique_child = node.children[0]
        if unique_child.level == num_levels - 1:
            return unique_child
        return MLTreeSearchFunctions.get_unique_fine_level_child(unique_child, num_levels)


class MLTreeModifier:
    def __init__(self, ground_proposal, subsampling_rates, rng_seed):
        self._ground_proposal = ground_proposal
        self._subsampling_rates = subsampling_rates
        self._rng = np.random.default_rng(rng_seed)

    def expand_tree(self, root):
        # Iterate over tree, add new nodes to computed accept leaves
        for node in root.leaves:
            if node.name == "a" and (node.logposterior is not None or node.computing):
                self._add_new_children_to_node(node)

        for node in root.leaves:
            if node.name == "r" and (
                (
                    len(node.parent.children) > 1
                    and (
                        util.leftsibling(node).logposterior is not None
                        or util.leftsibling(node).computing
                    )
                )
                or (len(node.parent.children) == 1)
            ):
                self._add_new_children_to_node(node)

    @staticmethod
    def update_probability_reached(root, acceptance_rate_estimator):
        for level_children in LevelOrderGroupIter(root):
            for node in level_children:
                acceptance_rate_estimate = acceptance_rate_estimator.get_acceptance_rate(node)
                
                if node.parent == None:
                    node.probability_reached = 1.0
                elif len(node.parent.children) == 1:
                    node.probability_reached = node.parent.probability_reached
                elif node.name == "a":
                    node.probability_reached = (
                        acceptance_rate_estimate * node.parent.probability_reached
                    )
                elif node.name == "r":
                    node.probability_reached = (
                        1 - acceptance_rate_estimate
                    ) * node.parent.probability_reached
                else:
                    raise ValueError(f"Invalid node name: {node.name}")

    @staticmethod
    def propagate_log_posterior_to_reject_children(root):
        for level_children in LevelOrderGroupIter(root):
            for child in level_children:
                same_level_parent = MLTreeSearchFunctions.get_same_level_parent(child)
                if child.name == "r" and same_level_parent is not None:
                    child.computing = same_level_parent.computing
                    child.logposterior = same_level_parent.logposterior

    def _add_new_children_to_node(self, node):
        accepted = MTNode("a", parent=node)
        rejected = MTNode("r", parent=node)

        for new_node in [accepted, rejected]:
            new_node.random_draw = self._rng.uniform()
        subsampling_rate = self._subsampling_rates[node.level]
        if (
            node.subchain_index == subsampling_rate - 1
        ):  # Reached end of subchain, go to next finer level
            for new_node in [accepted, rejected]:
                new_node.level = node.level + 1
                new_node.subchain_index = (
                    MLTreeSearchFunctions.get_same_level_parent(new_node).subchain_index + 1
                )
            accepted.state = node.state
            rejected.state = MLTreeSearchFunctions.get_same_level_parent(rejected).state
        else:  # Within subchain
            if node.level == 0:  # Extend subchain on coarsest level
                for new_node in [accepted, rejected]:
                    new_node.subchain_index = node.subchain_index + 1
                    new_node.level = node.level
                accepted.state = self._ground_proposal.propose(node.state)
                rejected.state = node.state
            else:  # Spawn new subchain on next coarser level
                for new_node in [accepted, rejected]:
                    new_node.subchain_index = 0
                    new_node.level = node.level - 1
                accepted.state = node.state
                rejected.state = node.state
                rejected.parent = None


class MLTreeVisualizer:
    pass
