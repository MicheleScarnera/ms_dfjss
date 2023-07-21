from typing import List, Any, Dict
import warnings
from collections import Counter
import time

import numpy as np

import dfjss_objects as dfjss
import dfjss_defaults as DEFAULTS
import dfjss_priorityfunction as pf
import dfjss_misc as misc

class GeneticAlgorithmSettings:
    def __init__(self):
        self.population_size = 200

        self.features = DEFAULTS.MANDATORY_NUMERIC_FEATURES
        self.operations = pf.DEFAULT_OPERATIONS.copy()
        #del self.operations["^"]

        self.tree_max_depth = 5

        self.tree_exnovo_generation_weights = {
            "another_branch": 0.43,
            "random_feature": 0.43,
            "random_number": 0.14
        }

        self.priority_function_random_number_range = (-4, 4)


class GeneticAlgorithm:
    settings: GeneticAlgorithmSettings

    def __init__(self, settings=None, rng_seed=None):
        if settings is None:
            settings = GeneticAlgorithmSettings()

        self.settings = settings

        self.rng = np.random.default_rng(seed=rng_seed)

        self.population = []

    def get_random_branch(self, current_depth=1):
        outcomes = list(self.settings.tree_exnovo_generation_weights.keys())
        weights = np.array(list(self.settings.tree_exnovo_generation_weights.values()))

        if current_depth == self.settings.tree_max_depth:
            for i, outcome in enumerate(outcomes):
                if outcome == "another_branch":
                    weights[i] = 0.
                    break

        left_weights = weights.copy()
        right_weights = weights.copy()

        operation = self.rng.choice(a=list(self.settings.operations.keys()))

        if operation == "^":
            left_weights[outcomes.index("random_number")] = 0.

            right_weights[outcomes.index("random_feature")] = 0.
            right_weights[outcomes.index("another_branch")] = 0.

        left_weights = left_weights / np.sum(left_weights)
        right_weights = right_weights / np.sum(right_weights)

        left_outcome = self.rng.choice(a=outcomes, p=left_weights)
        right_outcome = self.rng.choice(a=outcomes, p=right_weights)

        def realize_outcome(outcome):
            if outcome == "another_branch":
                return self.get_random_branch(current_depth=current_depth + 1)
            elif outcome == "random_feature":
                return self.rng.choice(a=self.settings.features)
            elif outcome == "random_number":
                return self.rng.uniform(low=self.settings.priority_function_random_number_range[0],
                                        high=self.settings.priority_function_random_number_range[1])
            else:
                raise ValueError(f"Unknown value in realize_outcome(outcome) (outcome == {outcome})")

        left_feature = realize_outcome(left_outcome)
        right_feature = realize_outcome(right_outcome)

        return pf.PriorityFunctionBranch(left_feature=left_feature,
                                         operation_character=operation,
                                         right_feature=right_feature)

    def get_random_individual(self):
        return pf.PriorityFunctionTree(features=self.settings.features,
                                       root_branch=self.get_random_branch())



