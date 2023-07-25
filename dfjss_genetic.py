from typing import List, Any, Dict
import warnings
from collections import Counter
import copy
import time

import numpy as np

import dfjss_objects as dfjss
import dfjss_defaults as DEFAULTS
import dfjss_priorityfunction as pf
import dfjss_misc as misc


class GeneticAlgorithmSettings:
    survival_rate: Any

    def __init__(self):
        self.population_size = 50
        self.total_steps = 100
        self.survival_rate = "knee-point"
        self.fitness_func = lambda objectives: objectives["mean_tardiness"] + 0.5 * objectives["mean_earliness"]

        self.number_of_simulations_per_individual = 3
        self.simulations_seeds = None

        self.features = DEFAULTS.MANDATORY_NUMERIC_FEATURES
        self.operations = pf.DEFAULT_OPERATIONS.copy()
        del self.operations["^"]

        self.tree_max_depth = 5

        self.priority_function_random_number_range = (-10, 10)
        self.random_number_decimal_places = 2

        self.tree_exnovo_generation_weights = {
            "another_branch": 0.43,
            "random_feature": 0.43,
            "random_number": 0.14
        }

        self.tree_mutation_weights = {
            "add_branch": 0.1,
            "remove_branch": 0.1,
            "change_feature": 0.4,
            "change_operation": 0.4
        }

        self.tree_mutation_changefeature_weights = {
            "random_feature": 0.8,
            "random_number": 0.2
        }

        self.tree_mutation_changeconstant_weights = {
            "random_feature": 0.2,
            "random_number": 0.8
        }

        self.warehouse_settings = dfjss.WarehouseSettings()


class GeneticAlgorithmRoutineOutput:
    best_individual: pf.PriorityFunctionTree
    best_fitness: float

    individuals_evaluated: int

    def __init__(self, best_individual=None, best_fitness=None):
        self.best_individual = best_individual
        self.best_fitness = best_fitness

        self.individuals_evaluated = 0


class GeneticAlgorithm:
    settings: GeneticAlgorithmSettings
    fitness_log: dict[str, float]

    def __init__(self, settings=None, rng_seed=None):
        if settings is None:
            settings = GeneticAlgorithmSettings()

        if settings.simulations_seeds is None:
            settings.simulations_seeds = np.linspace(0, 10000, num=settings.number_of_simulations_per_individual, endpoint=True, dtype=int)

        self.settings = settings

        self.rng = np.random.default_rng(seed=rng_seed)

        self.population = []

        self.fitness_log = dict()

    def random_number(self):
        return np.round(self.rng.uniform(low=self.settings.priority_function_random_number_range[0],
                                         high=self.settings.priority_function_random_number_range[1]),
                        decimals=self.settings.random_number_decimal_places)

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
                return self.random_number()
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

    def mutate_individual(self, individual, inplace=False):
        """

        :type inplace: bool
        :type individual: pf.PriorityFunctionTree
        """
        if not inplace:
            individual = individual.get_copy()

        outcomes = list(self.settings.tree_mutation_weights.keys())
        weights = np.array(list(self.settings.tree_mutation_weights.values()))

        if individual.depth() <= 1:
            weights[outcomes.index("remove_branch")] = 0.

        weights = weights / np.sum(weights)

        def is_feature(x):
            return x in self.settings.features or misc.is_number(x)

        outcome = self.rng.choice(a=outcomes, p=weights)
        crumbs = pf.representation_to_crumbs(repr(individual.root_branch), features=self.settings.features,
                                             operations=self.settings.operations)

        if outcome == "add_branch":
            features_indices = [i for i, crumb in enumerate(crumbs) if is_feature(crumb)]

            add_at = self.rng.choice(a=features_indices)

            crumbs.pop(add_at)

            crumbs.insert(add_at, "(")
            crumbs.insert(add_at + 1, self.rng.choice(a=self.settings.features))
            crumbs.insert(add_at + 2, self.rng.choice(a=list(self.settings.operations.keys())))
            crumbs.insert(add_at + 3, self.rng.choice(a=self.settings.features))
            crumbs.insert(add_at + 4, ")")

            individual.root_branch = pf.crumbs_to_root_branch(crumbs=crumbs)
        elif outcome == "remove_branch":
            # find a terminal branch and remove it
            par_locs = pf.crumbs_parenthesis_locations(crumbs)

            terminal_branches_indices = [par_loc
                                         for i, (par_loc, par) in enumerate(par_locs)
                                         if (i + 1 < len(par_locs)) and par == "(" and par_locs[i + 1][1] == ")"]

            remove_from = self.rng.choice(a=terminal_branches_indices)

            for _ in range(5):
                crumbs.pop(remove_from)

            random_feature = self.rng.choice(a=self.settings.features)
            crumbs.insert(remove_from, random_feature)

            individual.root_branch = pf.crumbs_to_root_branch(crumbs=crumbs)
        elif outcome == "change_feature":
            features_indices = [i for i, crumb in enumerate(crumbs) if is_feature(crumb)]

            change_at = self.rng.choice(a=features_indices)
            old_feature = crumbs[change_at]

            feature_outcomes = list(self.settings.tree_mutation_changefeature_weights.keys())
            if misc.is_number(old_feature):
                feature_weights = np.array(list(self.settings.tree_mutation_changeconstant_weights.values()))
            else:
                feature_weights = np.array(list(self.settings.tree_mutation_changefeature_weights.values()))

            feature_weights = feature_weights / np.sum(feature_weights)

            feature_outcome = self.rng.choice(a=feature_outcomes, p=feature_weights)
            new_feature = old_feature

            if feature_outcome == "random_feature":
                while new_feature == old_feature:
                    new_feature = self.rng.choice(a=self.settings.features)
            elif feature_outcome == "random_number":
                new_feature = self.random_number()

            crumbs[change_at] = new_feature

            individual.root_branch = pf.crumbs_to_root_branch(crumbs=crumbs)
        elif outcome == "change_operation":
            features_indices = [i for i, crumb in enumerate(crumbs) if crumb in self.settings.operations]

            change_at = self.rng.choice(a=features_indices)
            old_operation = crumbs[change_at]
            new_operation = old_operation
            while new_operation == old_operation:
                new_operation = self.rng.choice(a=list(self.settings.operations.keys()))

            crumbs[change_at] = new_operation

            individual.root_branch = pf.crumbs_to_root_branch(crumbs=crumbs)
        else:
            raise ValueError(f"Unknown outcome in mutate_individual ({outcome})")

        if not inplace:
            return individual

    def do_genetic_routine_once(self, verbose=0):
        result = GeneticAlgorithmRoutineOutput()

        if len(self.settings.simulations_seeds) != self.settings.number_of_simulations_per_individual:
            raise ValueError(f"Number of seeds provided in settings.simulations_seeds ({len(self.settings.simulations_seeds)}) is not the same as settings.number_of_simulations_per_individual ({self.settings.number_of_simulations_per_individual})")

        if len(self.population) == 0:
            self.population = [self.get_random_individual() for _ in range(self.settings.population_size)]

        fitness_values = np.zeros(shape=(len(self.population), self.settings.number_of_simulations_per_individual))
        start = time.time()
        for i, individual in enumerate(self.population):
            representation = repr(individual)
            if representation in self.fitness_log:
                fitness_values[i, :] = self.fitness_log[representation]
                continue
            else:
                result.individuals_evaluated += 1

            for j in range(self.settings.number_of_simulations_per_individual):
                if verbose > 1:
                    done = i * self.settings.number_of_simulations_per_individual + j
                    to_do = len(self.population) * self.settings.number_of_simulations_per_individual - 1

                    endch = ""
                    if done == to_do:
                        endch = "\n"

                    to_print = f"\rRunning simulations..."
                    if done > 0:
                        to_print += f" {done/to_do:.1%}, ETA {misc.timeformat_hhmmss(misc.timeleft(start, time.time(), done, to_do))}"

                    print(to_print, end=endch)

                if representation in self.fitness_log:
                    fitness_values[i, j] = self.fitness_log[representation]
                else:
                    porco_dio = False
                    if porco_dio:
                        fitness = self.rng.uniform(high=1000)
                    else:
                        seed = self.settings.simulations_seeds[j]
                        warehouse = dfjss.Warehouse(rng_seed=seed)
                        warehouse.settings = self.settings.warehouse_settings
                        warehouse.settings.decision_rule = pf.PriorityFunctionTreeDecisionRule(
                            priority_function_tree=individual
                        )

                        simulation_output = warehouse.simulate()
                        fitness = self.settings.fitness_func(simulation_output.get_objectives())

                    fitness_values[i, j] = fitness

        if verbose > 1:
            print(f"\tTook {misc.timeformat(time.time() - start)}")

        fitness_values = np.mean(fitness_values, axis=1)

        # annote fitness to precompute map
        for i in range(len(self.population)):
            representation = repr(self.population[i])

            if representation not in self.fitness_log:
                self.fitness_log[representation] = fitness_values[i]

        # sort by fitness
        fitness_order = np.argsort(a=fitness_values)

        population_amount_before = len(self.population)
        result.best_individual = self.population[fitness_order[0]]
        result.best_fitness = fitness_values[fitness_order[0]]

        if verbose > 1:
            print(f"\tBest individual: {result.best_individual}")
            print(f"\tBest fitness: {result.best_fitness:.2f}")

        cutoff_index_sorted = -1

        if self.settings.survival_rate == "knee-point":
            F = fitness_values[fitness_order[-1]]
            f = fitness_values[fitness_order[0]]
            I = len(self.population)

            a = - (F - f) / (I - 1)
            b = 1
            c = - f

            distances_from_funny_line = np.array([np.abs(a*i + b*fitness + c) / np.sqrt(a**2 + b**2) for i, fitness in enumerate(fitness_values[fitness_order])])
            distances_from_funny_line_order = np.argsort(distances_from_funny_line)
            cutoff_index_sorted = distances_from_funny_line_order[-1]

        elif misc.is_number(self.settings.survival_rate) and 0. < self.settings.survival_rate < 1.:
            cutoff_index_sorted = np.round(len(self.population) * self.settings.survival_rate)
        else:
            raise ValueError(f"self.settings.survival_rate has unexpected value {self.settings.survival_rate}")

        # remove unfit individuals
        self.population = [self.population[fitness_order[i]] for i in range(cutoff_index_sorted+1)]

        if verbose > 1:
            print(f"\t{(population_amount_before - len(self.population)) / population_amount_before:.1%} of the population was removed and repopulated")

        # repopulate
        repopulations_done = 0

        while len(self.population) < self.settings.population_size:
            new_individual = self.mutate_individual(
                individual=self.population[repopulations_done % population_amount_before],
                inplace=False)

            self.population.append(new_individual)

            repopulations_done += 1

        return result

    def run_genetic_algorithm(self, max_individuals_to_evaluate=-1, verbose=0):
        if verbose > 0:
            print("Running genetic algorithm...")

        individuals_evaluated_total = 0

        routine_output = None
        for step in range(1, self.settings.total_steps+1):
            if 0 < max_individuals_to_evaluate < individuals_evaluated_total:
                if verbose > 1:
                    print("Genetic algorithm evaluated maximum allowed number of individuals")
                break

            if verbose > 1:
                print(f"Step {step}")

            routine_output = self.do_genetic_routine_once(verbose=verbose)

            individuals_evaluated_total += routine_output.individuals_evaluated

        if verbose > 0:
            print("Done. Here is the best performing individual:")
            print(routine_output.best_individual)

        if verbose > 2:
            print("Fitness log:")
            print(misc.dictformat(self.fitness_log))