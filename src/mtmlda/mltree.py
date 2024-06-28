"""_summary_."""

import itertools
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing_extensions import Self

import anytree as atree
import anytree.exporter as exporter
import numpy as np


# ==================================================================================================
class BaseNode:
    """Markov tree node base class.

    The class contains all MCMC-related information for a node in the Markov tree.

    Attributes:
        probability_reached (float): The probability of reaching the node in the Markov tree
        state (np.ndarray): The state of the Markov chain associated with the node
        logposterior (float): The log-posterior probability of the state associated with the node
        logposterior_coarse (float): The log-posterior of the node state, but on the next coarser
            level of the model hierarchy
        computing (bool): Whether the log-posterior for the node's state is currently being computed
        random_draw (float): The random number used to decide whether to accept or reject the
            MCMC move associated with the node
        level (int): The level of the node in the multilevel hierarchy
        subchain_index (int): The index of the node within the subchain of its level
    """

    probability_reached: float = None
    state: np.ndarray = None
    logposterior: float = None
    logposterior_coarse: float = None
    computing: bool = False
    random_draw: float = None
    level: int = -1
    subchain_index: int = 0


class MTNode(BaseNode, atree.NodeMixin):
    """Markov tree node implementaiton.

    The class inherits MCMC-related information from '`BaseNode`' and is equiped with `Anytree`'s
    node functionality via a mixin.
    """

    def __init__(self, name: str, parent: Self = None, children: list[Self] = None):
        """Node constructor.

        Args:
            name (str): Identifier of the node
            parent (Self, optional): Paranet of the node. Defaults to None.
            children (list[Self], optional): Children of the node. Defaults to None.
        """
        super(BaseNode, self).__init__()
        self.name = name
        self.parent = parent
        if children:
            self.children = children


# ==================================================================================================
class MLTreeSearchFunctions:
    """Collection of functions for analyzing a Markov tree without modification.

    This class is only a namespace for the contained functions, all methods are static. The search
    functions iterate over an existing Markov tree and find nodes that fulfill certain criteria.

    Methods:
        find_max_probability_node: Find the node with the maximum probability of being reached
        get_same_level_parent: Find the parent node of the same level as the input node,
            if it exists
        get_unique_same_subchain_child: Find the unique child of a node with the same level, if it
            exists (i.e. if all intermdeiate MCMC decisions have been carried out)
        check_if_node_is_available_for_decision: Check if a node is available for an MCMC decision,
            depending on if its log-posterior and those of "adjacent nodes" have been computed
    """

    @staticmethod
    def find_max_probability_node(root: MTNode) -> MTNode:
        """Find node in the Markov tree with the maximum probability of being needed.

        More precisely, the method looks for the node that is most likely to be needed in the
        progression of the Markov tree. This is the node whose parent has the highest probability
        of being reached. The method only checks for accept nodes whose log-posterior has not been
        computed yet.

        Args:
            root (MTNode): Current root node of the Markov tree

        Returns:
            MTNode: Maximum probable node
        """
        max_probability = 0
        max_node = None

        for node in root.leaves:
            if node.name == "a":
                if node.parent is None:
                    parent_probability_reached = 1
                else:
                    parent_probability_reached = node.parent.probability_reached
                if (
                    parent_probability_reached > max_probability
                    and node.logposterior is None
                    and not node.computing
                ):
                    max_probability = parent_probability_reached
                    max_node = node

        assert max_node is not None, "No node found with maximum probability"
        return max_node

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def get_same_level_parent(node: MTNode) -> MTNode:
        """_summary_.

        Args:
            node (MTNode): _description_

        Returns:
            MTNode: _description_
        """
        current_candidate = node
        while (current_candidate := current_candidate.parent) is not None:
            if current_candidate.level == node.level:
                break
        return current_candidate

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def get_unique_same_subchain_child(node: MTNode) -> MTNode:
        """_summary_.

        Args:
            node (MTNode): _description_

        Returns:
            MTNode: _description_
        """
        current_candidates = node.children
        while True:
            if (len(current_candidates) != 1) or (current_candidates[0].level > node.level):
                return None
            elif current_candidates[0].level == node.level:
                return current_candidates[0]
            else:
                current_candidates = current_candidates[0].children

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def check_if_node_is_available_for_decision(node: MTNode) -> tuple[bool, bool, bool]:
        """_summary_.

        Args:
            node (MTNode): _description_

        Returns:
            tuple[bool, bool, bool]: _description_
        """
        decision_prerequisites_fulfilled = (
            node.name == "a"
            and node.parent is not None
            and len(node.parent.children) > 1
            and node.logposterior is not None
            and node.parent.logposterior is not None
        )

        if not decision_prerequisites_fulfilled:
            node_available_for_decision = False
            is_ground_level_decision = False
            is_two_level_decision = False
        else:
            same_level_parent = MLTreeSearchFunctions.get_same_level_parent(node)
            is_ground_level_decision = (node.level == 0) and (node.parent.level == 0)
            is_two_level_decision = (
                (node.level - 1 == node.parent.level)
                and (same_level_parent.logposterior is not None)
                and (same_level_parent.children[0].logposterior is not None)
            )
            node_available_for_decision = is_ground_level_decision or is_two_level_decision

        return node_available_for_decision, is_ground_level_decision, is_two_level_decision


# ==================================================================================================
class MLTreeModifier:
    def __init__(
        self,
        num_levels: int,
        ground_proposal: np.ndarray,
        subsampling_rates: Sequence[int],
        rng_seed: float,
    ) -> None:
        if not len(subsampling_rates) == num_levels:
            raise ValueError("Subsampling rates must be provided for all levels")
        self._num_levels = num_levels
        self._ground_proposal = ground_proposal
        self._subsampling_rates = subsampling_rates
        self._rng = np.random.default_rng(rng_seed)

    # ----------------------------------------------------------------------------------------------
    def expand_tree(self, root: MTNode) -> None:
        for node in root.leaves:
            if (node.logposterior is not None) or node.computing:
                self._add_new_children_to_node(node)
                self.update_descendants(node)

    # ----------------------------------------------------------------------------------------------
    def compress_resolved_subchains(self, root: MTNode) -> None:
        trying_to_compress = True

        while trying_to_compress:
            trying_to_compress = False

            for level_children in atree.LevelOrderGroupIter(root):
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

                    if trying_to_compress:
                        break
                if trying_to_compress:
                    break

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def update_descendants(root: MTNode) -> None:
        for level_children in itertools.islice(atree.LevelOrderGroupIter(root), 1, None):
            for child in level_children:
                same_level_parent = MLTreeSearchFunctions.get_same_level_parent(child)
                if same_level_parent is not None and child.state == same_level_parent.state:
                    child.computing = same_level_parent.computing
                    child.logposterior = same_level_parent.logposterior

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def discard_rejected_nodes(node: MTNode, accepted: bool) -> None:
        if accepted:
            atree.util.rightsibling(node).parent = None
        else:
            node.parent = None

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def update_probability_reached(root: MTNode, acceptance_rate_estimator: Any) -> None:
        for level_children in atree.LevelOrderGroupIter(root):
            for node in level_children:
                acceptance_rate_estimate = acceptance_rate_estimator.get_acceptance_rate(node.level)

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

    # ----------------------------------------------------------------------------------------------
    def _add_new_children_to_node(self, node: MTNode) -> None:
        accepted = MTNode(name="a", parent=node)
        rejected = MTNode(name="r", parent=node)

        for new_node in [accepted, rejected]:
            new_node.random_draw = self._rng.uniform(low=0, high=1, size=None)
        subsampling_rate = self._subsampling_rates[node.level]

        if node.subchain_index == subsampling_rate - 1:
            for new_node in [accepted, rejected]:
                new_node.level = node.level + 1
                same_level_parent = MLTreeSearchFunctions.get_same_level_parent(new_node)
                new_node.subchain_index = same_level_parent.subchain_index + 1

            rejected.state = same_level_parent.state
            accepted.state = node.state

        elif node.level == 0:
            for new_node in [accepted, rejected]:
                new_node.level = node.level
                new_node.subchain_index = node.subchain_index + 1

            rejected.state = node.state
            accepted.state = self._ground_proposal.propose(node.state)

        else:
            rejected.parent = None
            accepted.level = node.level - 1
            accepted.subchain_index = 0
            accepted.state = node.state

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
    def __init__(self, result_directory: Path = None) -> None:
        self._id_counter = 0
        self._result_dir = result_directory
        if self._result_dir is not None:
            os.makedirs(result_directory, exist_ok=True)

    # ----------------------------------------------------------------------------------------------
    def export_to_dot(self, mltree_root: MTNode) -> int:
        if self._result_dir is not None:
            tree_exporter = exporter.DotExporter(
                mltree_root, nodenamefunc=self._name_from_parents, nodeattrfunc=self._node_attr_func
            )
            tree_exporter.to_dotfile(self._result_dir / Path(f"mltree_{self._id_counter}.dot"))
            id_to_return = self._id_counter
            self._id_counter += 1
            return id_to_return
        else:
            return None

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
        name = node.name
        current_node = node
        while current_node := current_node.parent:
            name += current_node.name
        return name
