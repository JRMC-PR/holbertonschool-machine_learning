#!/usr/bin/env python3
"""Module for the Decision_Tree class."""
import numpy as np


class Node:
    """A class representing a node in a decision tree."""

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """Initialize a Node.

        Args:
            feature: The feature used for splitting at this node.
            threshold: The threshold value for the feature.
            left_child: The left child Node.
            right_child: The right child Node.
            is_root: A boolean indicating if this node is the root.
            depth: The depth of this node in the tree.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """Calculate the maximum depth of the tree below this node.

        Returns:
            The maximum depth of the tree below this node.
        """
        if self.is_leaf:
            return self.depth
        else:
            if self.left_child:
                left_depth = self.left_child.max_depth_below()
            else:
                left_depth = 0
            if self.right_child:
                right_depth = self.right_child.max_depth_below()
            else:
                right_depth = 0
            return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes below this node,
        optionally excluding non-leaf nodes.
        """
        if only_leaves:
            # If only leaves should be counted,
            # don't count this node
            count = self.left_child.count_nodes_below(only_leaves=True)
            count += self.right_child.count_nodes_below(only_leaves=True)
            return count
        else:
            # Otherwise, count this node plus the nodes below it
            count = 1 + self.left_child.count_nodes_below()
            count += self.right_child.count_nodes_below()
            return count

    def left_child_add_prefix(self, text):
        """ Add prefix to the left child """
        lines = text.split("\n")
        # Adding prefix to the first line
        new_text = "    +--" + lines[0] + "\n"
        # Adding prefix to the rest of the lines
        new_text += "\n".join(["    |  " + line for line in lines[1:-1]])
        # Append an additional newline character if there are multiple lines
        new_text += "\n" if len(lines) > 1 else ""
        return new_text

    def right_child_add_prefix(self, text):
        """ Add prefix to the right child """
        lines = text.split("\n")
        # Adding prefix to the first line
        new_text = "    +--" + lines[0] + "\n"
        # Adding prefix to the rest of the lines
        new_text += "\n".join(["     " + "  " + line for line in lines[1:-1]])
        # Append an additional newline character if there are multiple lines
        new_text += "\n" if len(lines) > 1 else ""
        return new_text

    def __str__(self):
        """
        Provides a string representation of the node and its children.

        Returns:
            str: A string representation of the subtree rooted at this node.
        """
        p = "root" if self.is_root else "-> node"
        result = f"{p} [feature={self.feature},\
 threshold={self.threshold}]\n"
        if self.left_child:
            result +=\
                self.left_child_add_prefix(self.left_child.__str__().strip())
        if self.right_child:
            result +=\
                self.right_child_add_prefix(self.right_child.__str__().strip())
        return result


class Leaf(Node):
    """A class representing a leaf node in a decision tree."""

    def __init__(self, value, depth=None):
        """Initialize a Leaf.

        Args:
            value: The value or class label that
            is predicted at this leaf node.
            depth: The depth of this node in the tree.
        """
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """Calculate the maximum depth of the tree below this node.

        Since this is a leaf node, it returns its own depth.

        Returns:
            The depth of this leaf node.
        """
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """This method counts the number of nodes below this node."""
        return 1

    def __str__(self):
        """Return a string representation of the node."""
        return (f"-> leaf [value={self.value}] ")


class Decision_Tree():
    """A class representing a decision tree."""

    def __init__(self, max_depth=10, min_pop=1,
                 seed=0, split_criterion="random", root=None):
        """Initialize a Decision_Tree.

        Args:
            max_depth: The maximum depth of the tree.
            min_pop: The minimum population size at a
            node for a split to be considered.
            seed: The seed for the random number generator.
            split_criterion: The criterion used for splitting at each node.
            root: The root node of the tree.
        """
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """Calculate the maximum depth of the tree.

        Returns:
            The maximum depth of the tree.
        """
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """Count the number of nodes in the tree.

        Args:
            only_leaves: A boolean indicating
            if only leaf nodes should be counted.

        Returns:
            The number of nodes in the tree.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """Return a string representation of the tree."""
        return self.root.__str__()
