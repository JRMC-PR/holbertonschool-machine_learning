# Desition Trees


## Learning Objectives
---
This project aims to implement decision trees from scratch. It is important for engineers to understand how the tools we use are built for two reasons. First, it gives us confidence in our skills. Second, it helps us when we need to build our own tools to solve unsolved problems.
The first three references point to historical papers where the concepts were first studied.
References 4 to 9 can help if you feel you need some more explanation about the way we split nodes.
William A. Belson is usually credited for the invention of decision trees (read reference 11).
Despite our efforts to make it efficient, we cannot compete with Sklearn’s implementations (since they are done in C). In real life, it is thus recommended to use Sklearn’s tools.
In this regard, it is warmly recommended to watch the video referenced as (10) above. It shows how to use Sklearn’s decision trees and insists on the methodology.
---




#  Tasks
### We will progressively add methods in the following 3 classes :
---
class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature                  = feature
        self.threshold                = threshold
        self.left_child               = left_child
        self.right_child              = right_child
        self.is_leaf                  = False
        self.is_root                  = is_root
        self.sub_population           = None
        self.depth                    = depth

class Leaf(Node):
    def __init__(self, value, depth=None) :
        super().__init__()
        self.value   = value
        self.is_leaf = True
        self.depth   = depth

class Decision_Tree() :
    def __init__(self, max_depth=10, min_pop=1, seed=0,split_criterion="random", root=None) :
        self.rng               = np.random.default_rng(seed)
        if root :
            self.root          = root
        else :
            self.root          = Node(is_root=True)
        self.explanatory       = None
        self.target            = None
        self.max_depth         = max_depth
        self.min_pop           = min_pop
        self.split_criterion   = split_criterion
        self.predict           = None

Once built, decision trees are binary trees : a node either is a leaf or has two children. It never happens that a node for which is_leaf is False has its left_child or right_child left unspecified.
The first three tasks are a warm-up designed to review the basics of class inheritance and recursion (nevertheless, the functions coded in these tasks will be reused in the rest of the project).
Our first objective will be to write a Decision_Tree.predict method that takes the explanatory features of a set of individuals and returns the predicted target value for these individuals.
Then we will write a method Decision_Tree.fit that takes the explanatory features and the targets of a set of individuals, and grows the tree from the root to the leaves to make it in an efficient prediction tool.
Once these tasks will be accomplished, we will introduce a new class Random_Forest that will also be a powerful prediction tool.
Finally, we will write a variation on Random_Forest, called Isolation_Random_forest, that will be a tool to detect outliers
---

## 0. Depth of a decision tree
All the nodes of a decision tree have their depth attribute. The depth of the root is 0 , while the children of a node at depth k have a depth of k+1. We want to find the maximum of the depths of the nodes (including the leaves) in a decision tree. In order to do so, we added a method def depth(self): in the Decision_Treeclass, a method def max_depth_below(self): in the Leaf class.

Task: Update the class Node by adding the method def max_depth_below(self):.

Down below is the content of the file 0-build_decision_tree.py .

---
#!/usr/bin/env python3

import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left_child=None, right_child=None, is_root=False, depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self) :

            ####### FILL IN THIS METHOD

class Leaf(Node):
    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self) :
        return self.depth

class Decision_Tree():
    def __init__(self, max_depth=10, min_pop=1, seed=0, split_criterion="random", root=None):
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

    def depth(self) :
        return self.root.max_depth_below()
---

### Main to test your work
---
0-main.py
#!/usr/bin/env python3

Node = __import__('0-build_decision_tree').Node
Leaf = __import__('0-build_decision_tree').Leaf
Decision_Tree = __import__('0-build_decision_tree').Decision_Tree

def example_0() :
    leaf0         = Leaf(0, depth=1)
    leaf1         = Leaf(0, depth=2)
    leaf2         = Leaf(1, depth=2)
    internal_node = Node( feature=1, threshold=30000, left_child=leaf1, right_child=leaf2,          depth=1 )
    root          = Node( feature=0, threshold=.5   , left_child=leaf0, right_child=internal_node , depth=0 , is_root=True)
    return Decision_Tree(root=root)


def example_1(depth):
    level = [Leaf(i, depth=depth) for i in range(2 ** depth)]
    level.reverse()

    def get_v(node):
        if node.is_leaf:
            return node.value
        else:
            return node.threshold

    for d in range(depth):
        level = [Node(feature=0,
                      threshold=(get_v(level[2 * i]) + get_v(level[2 * i + 1])) / 2,
                      left_child=level[2 * i],
                      right_child=level[2 * i + 1], depth=depth - d - 1) for i in range(2 ** (depth - d - 1))]
    root = level[0]
    root.is_root = True
    return Decision_Tree(root=root)

print(example_0().depth())
print(example_1(5).depth())
---
