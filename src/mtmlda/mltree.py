import os
from collections.abc import Sequence
from pathlib import Path
from typing_extensions import Any

import numpy as np
import anytree as at
import anytree.exporter as exporter


# ==================================================================================================
class MTNodeBase:
    probability_reached: float = None
    state: float = None
    logposterior: float = None
    logposterior_coarse: float = None
    computing: bool = False
    random_draw: float = None
    level: int = -1
    subchain_index: int = 0


class MTNode(MTNodeBase, at.NodeMixin):
    def __init__(self, name: str, parent: Path = None, children: Path = None):
        super(MTNodeBase, self).__init__()
        self.name = name
        self.parent = parent
        if children:
            self.children = children


# ==================================================================================================
class MLTreeSearchFunctions:
    @staticmethod
    def find_max_probability_node(root: MTNode) -> MTNode:
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

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def get_same_level_parent(node: MTNode) -> MTNode:
        if node.parent is None:
            return None
        same_level_parent = node.parent
        while same_level_parent.level != node.level:
            if same_level_parent.parent is None:
                return None
            same_level_parent = same_level_parent.parent
        return same_level_parent

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def get_unique_fine_level_child(node: MTNode, num_levels: int) -> MTNode:
        if len(node.children) != 1:
            return None
        unique_child = node.children[0]
        if unique_child.level == num_levels - 1:
            return unique_child
        return MLTreeSearchFunctions.get_unique_fine_level_child(unique_child, num_levels)

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def get_unique_same_subchain_child(node: MTNode) -> MTNode:
        iter = node
        while True:
            if len(iter.children) != 1:
                return None
            iter = iter.children[0]
            if iter.level > node.level:  # we have left the subchain
                return None
            if iter.level == node.level and iter is not node:
                return iter


# ==================================================================================================
class MLTreeModifier:
    def __init__(
        self,
        num_levels: int,
        ground_proposal: np.ndarray,
        subsampling_rates: Sequence[int],
        rng_seed: float,
    ) -> None:
        self._num_levels = num_levels
        self._ground_proposal = ground_proposal
        self._subsampling_rates = subsampling_rates
        self._rng = np.random.default_rng(rng_seed)

    # ----------------------------------------------------------------------------------------------
    def expand_tree(self, root: MTNode) -> None:
        # Iterate over tree, add new nodes to computed accept leaves
        for node in root.leaves:
                
            if node.name == "a" and (node.logposterior is not None or node.computing):
                self._add_new_children_to_node(node)

            if node.name == "r" and (
                (node.parent is None)
                or (len(node.parent.children) == 1)
                or
                (
                    len(node.parent.children) > 1
                    and (
                        at.util.leftsibling(node).logposterior is not None
                        or at.util.leftsibling(node).computing
                    )
                )   
            ):
                self._add_new_children_to_node(node)

    def compress_resolved_subchains(self, root: MTNode) -> None:
        trying_to_compress = True
        while trying_to_compress:
            trying_to_compress = False

            for level_children in at.LevelOrderGroupIter(root):
                for node in level_children:
                    if node.level == self._num_levels - 1:
                        continue
                    same_subchain_child = MLTreeSearchFunctions.get_unique_same_subchain_child(node)
                    if same_subchain_child is None:
                        continue
                    same_subchain_grandchild = MLTreeSearchFunctions.get_unique_same_subchain_child(
                        same_subchain_child
                    )
                    if same_subchain_grandchild is None:
                        continue

                    node.children[0].parent = None
                    same_subchain_grandchild.parent = node
                    trying_to_compress = True

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def update_probability_reached(root: MTNode, acceptance_rate_estimator: Any) -> None:
        for level_children in at.LevelOrderGroupIter(root):
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

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def propagate_log_posterior_to_reject_children(root: MTNode) -> None:
        for level_children in at.LevelOrderGroupIter(root):
            for child in level_children:
                same_level_parent = MLTreeSearchFunctions.get_same_level_parent(child)
                if child.name == "r" and same_level_parent is not None:
                    child.computing = same_level_parent.computing
                    child.logposterior = same_level_parent.logposterior

    # ----------------------------------------------------------------------------------------------
    def _add_new_children_to_node(self, node: MTNode) -> None:
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

    # ----------------------------------------------------------------------------------------------
    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    # ----------------------------------------------------------------------------------------------
    @rng.setter
    def rng(self, rng: np.random.Generator) -> None:
        self._rng = rng


# ==================================================================================================
class MLTreeVisualizer:
    _base_size = 1
    _fixed_size = True
    _style = "filled"
    _shape = "circle"
    _border_color = "slategray4"
    _color_not_visited = " 	azure2"
    _color_computing = "darkgoldenrod1"
    _color_visited = "mediumaquamarine"

    # ----------------------------------------------------------------------------------------------
    def __init__(self, result_directory_path: Path = None) -> None:
        self._result_dir = result_directory_path
        self._print_counter = 0
        if self._result_dir is not None:
            os.makedirs(result_directory_path, exist_ok=True)

    # ----------------------------------------------------------------------------------------------
    def export_to_dot(self, mltree_root: MTNode) -> None:
        if self._result_dir is not None:
            tree_exporter = exporter.DotExporter(
                mltree_root, nodenamefunc=self._name_from_parents, nodeattrfunc=self._node_attr_func
            )
            tree_exporter.to_dotfile(self._result_dir / Path(f"mltree_{self._print_counter}.dot"))
            self._print_counter += 1

    # ----------------------------------------------------------------------------------------------
    @classmethod
    def _node_attr_func(cls, node: MTNode) -> str:
        node_size = (1 + 0.75 * node.level) * cls._base_size
        if node.logposterior is not None:
            color = cls._color_visited
        elif node.computing:
            color = cls._color_computing
        else:
            color = cls._color_not_visited
        node_label = (
            f"N: {node.name}\n"
            f"PR: {node.probability_reached:.3f}\n"
            f"LVL: {node.level}\n"
            f"SCI: {node.subchain_index}"
        )

        attr_string = (
            f"""label="{node_label}", """
            f"shape={cls._shape}, "
            f"fixedsize={cls._fixed_size}, "
            f"bordercolor={cls._border_color}, "
            f"style={cls._style}, "
            f"fillcolor={color}, "
            f"width={node_size}, "
            f"height={node_size}"
        )
        return attr_string

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _name_from_parents(node: MTNode) -> str:
        if node.parent is None:
            return node.name
        else:
            return MLTreeVisualizer._name_from_parents(node.parent) + node.name
