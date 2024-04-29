#!/usr/bin/env python3
"""
Module implementing Isolation Random Forest for outlier detection using
Isolation Trees. Designed for high-dimensional datasets, it identifies
anomalies based on data splits by feature selection.
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
Isolation_Random_Tree = __import__('10-isolation_tree').Isolation_Random_Tree


class Isolation_Random_Forest:
    """ Defines an Isolation Random Forest. """

    def __init__(self, n_trees=100, max_depth=10, min_pop=1, seed=0):
        """ Initializes the Isolation Random Forest. """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed
        self.numpy_preds = []

    def _fit_tree(self, seed, explanatory):
        """ Helper method to fit a single tree. """
        tree = Isolation_Random_Tree(max_depth=self.max_depth, seed=seed)
        tree.fit(explanatory)
        return (tree.predict, tree.depth(), tree.count_nodes(),
                tree.count_nodes(only_leaves=True))

    def fit(self, explanatory, n_trees=100, verbose=0):
        """ Fits model to training data using parallel processing. """
        self.explanatory = explanatory
        self.numpy_preds = []
        depths = []
        nodes = []
        leaves = []

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda i: self._fit_tree(self.seed + i, explanatory),
                range(n_trees)))

        for result in results:
            self.numpy_preds.append(result[0])
            depths.append(result[1])
            nodes.append(result[2])
            leaves.append(result[3])

        if verbose == 1:
            print(f"Training finished.\n"
                  f"  - Mean depth: {np.mean(depths)}\n"
                  f"  - Mean number of nodes: {np.mean(nodes)}\n"
                  f"  - Mean number of leaves: {np.mean(leaves)}")

    def predict(self, explanatory):
        """ Makes predictions for a given set of examples. """
        predictions = np.array([f(explanatory) for f in self.numpy_preds])
        return np.mean(predictions, axis=0)

    def suspects(self, explanatory, n_suspects):
        """ Returns top n suspects with the smallest depths. """
        depths = self.predict(explanatory)
        sorted_indices = np.argsort(depths)
        suspect_data = explanatory[sorted_indices[:n_suspects]]
        suspect_depths = depths[sorted_indices[:n_suspects]]
        return suspect_data, suspect_depths
