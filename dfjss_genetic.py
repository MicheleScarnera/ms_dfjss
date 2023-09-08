from typing import List, Any, Dict
import warnings
from collections import Counter
import copy
import time
import datetime
import json
from pathlib import Path
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd

import dfjss_objects as dfjss
import dfjss_defaults as DEFAULTS
import dfjss_priorityfunction as pf
import dfjss_phenotype as pht
import dfjss_misc as misc

FITNESSLOG_CSV_NAME = "fitness_log"
GENALGOLOG_CSV_NAME = "genalgo_log"
LASTSTATE_JSON_NAME = "last_state"

def init_pool_processes(the_lock):
    '''Initialize each process with a global variable lock.
    '''
    global lock
    lock = the_lock


class GeneticAlgorithmSettings:
    reproduction_rate: Any

    def __init__(self):
        self.population_size = 50
        self.tournament_percent_size = 0.25
        self.total_steps = 100

        self.crossover_rate = 0.9
        self.crossover_rate_increment = 0.

        self.reproduction_rate = 0.08
        self.reproduction_rate_increment = 0.

        self.mutation_rate = 0.02
        self.mutation_rate_increment = 0.

        self.fitness_is_random = False

        self.number_of_simulations_per_individual = 3
        self.simulations_reduce = np.mean
        self.simulations_seeds = None

        self.features = DEFAULTS.MANDATORY_NUMERIC_FEATURES
        self.operations = pf.DEFAULT_OPERATIONS.copy()
        del self.operations["^"]

        self.tree_generation_max_depth = 5
        self.tree_generation_mode = "half_and_half"

        self.tree_transformation_max_depth = 8

        self.priority_function_random_number_range = (-10, 10)
        self.random_number_granularity = 0.25

        self.tree_exnovo_generation_weights = {
            "another_branch": 0.43,
            "random_feature": 0.43,
            "random_number": 0.14
        }

        self.tree_mutation_weights = {
            "add_branch": 0.1,
            "remove_branch": 0.1,
            "hoist_branch": 0.1,
            "change_feature": 0.35,
            "change_operation": 0.35,
            "do_nothing": 0
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

        self.save_logs = True
        self.fitness_log_is_phenotype_mapper = True
        self.phenotype_mapper_scenarios_amount = 16
        self.phenotype_exploration_attempts_during_crossover = 3
        self.depth_attempts_during_crossover = 3

        self.multiprocessing_processes = 4

    def fitness_func(self, objectives):
        return 0.8 * (0.75 * objectives["mean_tardiness"] + 0.25 * objectives["max_tardiness"]) +\
               0.2 * (0.75 * objectives["mean_earliness"] + 0.25 * objectives["max_earliness"])


class GeneticAlgorithmRoutineOutput:
    best_individual: pf.PriorityFunctionTree
    best_fitness: float

    population_data: dict[str, float]

    individuals_evaluated: int

    def __init__(self, best_individual=None, best_fitness=None, population_data=None):
        self.best_individual = best_individual
        self.best_fitness = best_fitness

        self.population_data = population_data if population_data is not None else dict()

        self.individuals_evaluated = 0


class GeneticAlgorithm:
    settings: GeneticAlgorithmSettings

    # fitness_log: dict[str, float]

    def __init__(self, settings=None, rng_seed=None):
        if settings is None:
            settings = GeneticAlgorithmSettings()

        if settings.simulations_seeds is None:
            settings.simulations_seeds = np.linspace(0, 10000, num=settings.number_of_simulations_per_individual,
                                                     endpoint=True, dtype=int)

        self.settings = settings

        self.rng = np.random.Generator(np.random.PCG64(seed=rng_seed))

        self.population = []

        if self.settings.fitness_log_is_phenotype_mapper:
            self.fitness_log = pht.PhenotypeMapper(scenarios_seed=rng_seed,
                                                   reference_scenarios_amount=self.settings.phenotype_mapper_scenarios_amount)
        else:
            self.fitness_log = dict()

        # multiprocessing

        self.__worker_tasks = []

    def random_number(self):
        return np.round(self.rng.uniform(low=self.settings.priority_function_random_number_range[0],
                                         high=self.settings.priority_function_random_number_range[1])
                        / self.settings.random_number_granularity) * self.settings.random_number_granularity

    def get_random_branch(self, generation_mode="short", current_depth=1, is_generation=True):
        outcomes = list(self.settings.tree_exnovo_generation_weights.keys())
        weights = np.array(list(self.settings.tree_exnovo_generation_weights.values()))

        max_depth = self.settings.tree_generation_max_depth if is_generation else self.settings.tree_transformation_max_depth

        if current_depth == max_depth:
            weights[outcomes.index("another_branch")] = 0.
        elif generation_mode == "wide":
            weights[outcomes.index("random_number")] = 0.
            weights[outcomes.index("random_feature")] = 0.

        left_weights = weights.copy()
        right_weights = weights.copy()

        if generation_mode == "long":
            if current_depth < max_depth:
                if self.rng.uniform() < 0.5:
                    left_weights[outcomes.index("random_number")] = 0.
                    left_weights[outcomes.index("random_feature")] = 0.
                else:
                    right_weights[outcomes.index("random_number")] = 0.
                    right_weights[outcomes.index("random_feature")] = 0.

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
                return self.get_random_branch(current_depth=current_depth + 1, is_generation=False)
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
        if self.settings.tree_generation_mode == "half_and_half":
            gen = self.rng.choice(a=["long", "wide"])
        else:
            gen = self.settings.tree_generation_mode

        return pf.PriorityFunctionTree(features=self.settings.features,
                                       root_branch=self.get_random_branch(generation_mode=gen))

    def combine_individuals(self, individual_1, individual_2, verbose=0):
        repr_1 = repr(individual_1)
        repr_2 = repr(individual_2)

        if verbose > 2:
            print(f"Individual 1: {repr_1}")
            print(f"Individual 2: {repr_2}")

        crumbs_1 = pf.representation_to_crumbs(repr_1, features=self.settings.features)
        crumbs_2 = pf.representation_to_crumbs(repr_2, features=self.settings.features)

        is_valid_node = lambda x: x == "(" or x in self.settings.features or misc.is_number(x)

        # choose a random node on each individual (using open parentheses or features)
        crumb_i_chosen_1 = self.rng.choice(a=[i for i, crumb in enumerate(crumbs_1) if is_valid_node(crumb) and i > 0])
        crumb_i_chosen_2 = self.rng.choice(a=[i for i, crumb in enumerate(crumbs_2) if is_valid_node(crumb)])

        crumb_chosen_1_is_feature = is_valid_node(crumbs_1[crumb_i_chosen_1]) and crumbs_1[crumb_i_chosen_1] != "("
        crumb_chosen_2_is_feature = is_valid_node(crumbs_2[crumb_i_chosen_2]) and crumbs_2[crumb_i_chosen_2] != "("

        # get "appendage" (thing to add to individual 1, taken from individual 2)
        appendage = []
        appendage_no_open = 0
        appendage_no_closed = 0

        for j in range(crumb_i_chosen_2, len(crumbs_2)):
            crumb = crumbs_2[j]

            if crumb == "(":
                appendage_no_open += 1
            elif crumb == ")":
                appendage_no_closed += 1

            appendage.append(crumb)

            if (
                    not crumb_chosen_2_is_feature and 0 < appendage_no_open == appendage_no_closed) or crumb_chosen_2_is_feature:
                break

        # replace part of individual 1 with the appendage
        for j in range(crumb_i_chosen_1, len(crumbs_1)):
            crumb = crumbs_1.pop(crumb_i_chosen_1)

            if crumb == "(":
                appendage_no_open += 1
            elif crumb == ")":
                appendage_no_closed += 1

            if (
                    not crumb_chosen_1_is_feature and 0 < appendage_no_open == appendage_no_closed) or crumb_chosen_1_is_feature:
                break

        crumbs_1[crumb_i_chosen_1:crumb_i_chosen_1] = appendage

        result = pf.PriorityFunctionTree(root_branch=pf.crumbs_to_root_branch(crumbs_1),
                                         features=self.settings.features,
                                         operations=self.settings.operations)

        if verbose > 2:
            print(f"Crossover: {result}")

        return result

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
            weights[outcomes.index("hoist_branch")] = 0.

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
            # find an inner branch, remove it and replace it with a random feature
            par_locs = pf.crumbs_parenthesis_locations(crumbs)

            branches_indices = [par_loc
                                for i, (par_loc, par) in enumerate(par_locs)
                                if i > 0 and par == "("] # if (i + 1 < len(par_locs)) and par == "(" and par_locs[i + 1][1] == ")"

            remove_from = self.rng.choice(a=branches_indices)

            open_removed = 0
            closed_removed = 0

            while True:
                removed_crumb = crumbs.pop(remove_from)

                if removed_crumb == "(":
                    open_removed += 1
                elif removed_crumb == ")":
                    closed_removed += 1

                if open_removed > 0 and open_removed == closed_removed:
                    break

            random_feature = self.rng.choice(a=self.settings.features)
            crumbs.insert(remove_from, random_feature)

            individual.root_branch = pf.crumbs_to_root_branch(crumbs=crumbs)
        elif outcome == "hoist_branch":
            # find an inner branch and make it the whole individual
            par_locs = pf.crumbs_parenthesis_locations(crumbs)

            branches_indices = [par_loc
                                for i, (par_loc, par) in enumerate(par_locs)
                                if
                                i > 0 and par == "("]  # if (i + 1 < len(par_locs)) and par == "(" and par_locs[i + 1][1] == ")"

            remove_from = self.rng.choice(a=branches_indices)

            open_removed = 0
            closed_removed = 0
            new_crumbs = []

            while True:
                removed_crumb = crumbs.pop(remove_from)

                new_crumbs.append(removed_crumb)

                if removed_crumb == "(":
                    open_removed += 1
                elif removed_crumb == ")":
                    closed_removed += 1

                if open_removed > 0 and open_removed == closed_removed:
                    break

            individual.root_branch = pf.crumbs_to_root_branch(crumbs=new_crumbs)
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
        elif outcome == "do_nothing":
            pass
        else:
            raise ValueError(f"Unknown outcome in mutate_individual \"{outcome}\"")

        if not inplace:
            return individual

    def import_state(self, folder_name):
        json_filepath = f"{folder_name}/{LASTSTATE_JSON_NAME}.json"
        with open(json_filepath) as file:
            data = json.load(file)

            self.rng.bit_generator.state = data["rng_state"]

            self.population = [pf.representation_to_priority_function_tree(representation=individual_repr,
                                                                           features=self.settings.features,
                                                                           operations=self.settings.operations)
                               for individual_repr in data["population"]]

    def do_worker_task(self, task_tuple, is_multiprocessing=True):
        individual_index, seed_index, individual, seed, warehouse_settings = task_tuple

        representation = repr(individual)

        if representation in self.fitness_log:
            return self.fitness_log[representation]

        if self.settings.fitness_is_random:
            #fitness = np.mean(self.rng.uniform(high=1000, size=3))
            fitness, _ = np.modf(time.time())
            fitness *= 1000.
        else:
            warehouse = dfjss.Warehouse(rng_seed=seed, settings=warehouse_settings)

            decision_rule = pf.PriorityFunctionTreeDecisionRule(
                priority_function_tree=individual
            )

            warehouse_settings.decision_rule = decision_rule

            #print(warehouse.settings.generation_simulation_ranges["simulation_random_job_arrival_rate"])

            simulation_output = warehouse.simulate()

            fitness = self.settings.fitness_func(simulation_output.get_objectives())

        return fitness

    def do_genetic_routine_once(self, current_step, max_steps, verbose=0):
        result = GeneticAlgorithmRoutineOutput()

        if len(self.settings.simulations_seeds) != self.settings.number_of_simulations_per_individual:
            raise ValueError(
                f"Number of seeds provided in settings.simulations_seeds ({len(self.settings.simulations_seeds)}) is not the same as settings.number_of_simulations_per_individual ({self.settings.number_of_simulations_per_individual})")

        # if population is empty (or below desired amount) fill with random individuals
        while len(self.population) < self.settings.population_size:
            # new individuals must be never seen before
            while True:
                random_individual = self.get_random_individual()
                repr_individual = repr(random_individual)

                if repr_individual in self.fitness_log or repr_individual in [repr(individual) for individual in
                                                                              self.population]:
                    continue
                else:
                    break

            self.population.append(random_individual)

        # compute fitness values
        fitness_values = np.zeros(shape=(len(self.population), self.settings.number_of_simulations_per_individual))

        self.__worker_tasks = []
        for individual_index in range(len(self.population)):
            for seed_index in range(self.settings.number_of_simulations_per_individual):
                self.__worker_tasks.append((individual_index, seed_index, self.population[individual_index], self.settings.simulations_seeds[seed_index], copy.copy(self.settings.warehouse_settings)))

        # shuffle the task order to make the ETA estimation less biased
        self.rng.shuffle(x=self.__worker_tasks)

        start = time.time()

        if self.settings.multiprocessing_processes > 1:
            lock = mp.Lock()
            with mp.get_context("spawn").Pool(initializer=init_pool_processes, initargs=(lock,), processes=self.settings.multiprocessing_processes) as pool:
            #with ThreadPool(initializer=init_pool_processes, initargs=(lock,), processes=self.settings.multiprocessing_processes) as pool:
                apply_results = []
                for task in self.__worker_tasks:
                    apply_results.append(pool.apply_async(func=self.do_worker_task, args=(task, True)))

                pool.close()

                if verbose > 1:
                    initial_tasks = len(self.__worker_tasks)
                    previous_tasks_done = 0
                    current_eta = -1
                    time_increment = 1
                    while True:
                        time.sleep(time_increment)

                        tasks_done = np.sum([apply_result.ready() for apply_result in apply_results])

                        completed = tasks_done >= initial_tasks

                        if tasks_done > previous_tasks_done:
                            current_eta = misc.timeleft(start, time.time(), tasks_done, initial_tasks)
                        else:
                            current_eta = max(current_eta - time_increment, 0)

                        endch = ""
                        if completed:
                            endch = "\n"

                        to_print = f"\rRunning simulations..."
                        if tasks_done > 0:
                            absolute_eta = datetime.datetime.now() + datetime.timedelta(seconds=current_eta)
                            to_print += f" {tasks_done} / {initial_tasks} ({tasks_done / initial_tasks:.1%}), ETA {misc.timeformat_hhmmss(current_eta)} ({absolute_eta.strftime('%a %d %b %Y, %H:%M:%S')})"

                        previous_tasks_done = tasks_done

                        if completed:
                            break

                        print(to_print, end=endch, flush=True)

                pool.join()

                for (ind, seed, *_), apply_result in zip(self.__worker_tasks, apply_results):
                    fitness_values[ind, seed] = apply_result.get()
        else:
            initial_tasks = len(self.__worker_tasks)
            for i, task in enumerate(self.__worker_tasks):
                individual_index, seed_index, *_ = task

                fitness_values[individual_index, seed_index] = self.do_worker_task(task_tuple=task, is_multiprocessing=False)

                if verbose > 1:
                    tasks_done = i

                    completed = tasks_done >= initial_tasks

                    endch = ""
                    if completed:
                        endch = "\n"

                    to_print = f"\rRunning simulations..."
                    if tasks_done > 0:
                        to_print += f" {tasks_done} / {initial_tasks} ({tasks_done / initial_tasks:.1%}), ETA {misc.timeformat_hhmmss(misc.timeleft(start, time.time(), tasks_done, initial_tasks))}"

                    print(to_print, end=endch)

        if verbose > 1:
            print(f"\tTook {misc.timeformat(time.time() - start)}")

        fitness_values = self.settings.simulations_reduce(fitness_values, axis=1)

        result.population_data = pd.DataFrame(
            {"Individual": [repr(individual) for individual in self.population],
             "Fitness": fitness_values}
        )

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

            print(
                f"\tMean fitness: {np.mean(fitness_values):.2f} (Interquartile range: [{np.quantile(fitness_values, q=0.25):.2f}, {np.quantile(fitness_values, q=0.75):.2f}])")

        cutoff_index_sorted = -1

        if self.settings.reproduction_rate == "knee-point":
            F = fitness_values[fitness_order[-1]]
            f = fitness_values[fitness_order[0]]
            I = len(self.population)

            a = - (F - f) / (I - 1)
            b = 1
            c = - f

            distances_from_funny_line = np.array(
                [np.abs(a * i + b * fitness + c) / np.sqrt(a ** 2 + b ** 2) for i, fitness in
                 enumerate(fitness_values[fitness_order])])
            distances_from_funny_line_order = np.argsort(distances_from_funny_line)
            cutoff_index_sorted = distances_from_funny_line_order[-1]

        elif misc.is_number(self.settings.reproduction_rate) and 0. < self.settings.reproduction_rate < 1.:
            cutoff_index_sorted = int(np.round(len(self.population) * self.settings.reproduction_rate))
        else:
            raise ValueError(f"self.settings.reproduction_rate has unexpected value {self.settings.reproduction_rate}")

        # wipe current population
        old_population = self.population
        old_population_sorted = [self.population[fitness_order[i]] for i in range(len(fitness_order))]
        self.population = []

        # fitness weights (to be used in "lower fitness is better" random draws
        fitness_weights = np.array([1. / (i + 1) for i in range(population_amount_before)])
        fitness_weights = fitness_weights / np.sum(fitness_weights)

        # reproduction, crossover, mutation
        reproducing_individuals_amount = cutoff_index_sorted

        current_reproduction_rate = max(0.01,
                                        reproducing_individuals_amount / population_amount_before + self.settings.reproduction_rate_increment * (
                                                    current_step - 1))
        current_crossover_rate = max(0.01,
                                     self.settings.crossover_rate + self.settings.crossover_rate_increment * (
                                                 current_step - 1))
        current_mutation_rate = max(0.01,
                                    self.settings.mutation_rate + self.settings.mutation_rate_increment * (
                                                current_step - 1))

        rates_norm = current_reproduction_rate + current_crossover_rate + current_mutation_rate

        current_reproduction_rate, current_crossover_rate, current_mutation_rate = current_reproduction_rate / rates_norm, current_crossover_rate / rates_norm, current_mutation_rate / rates_norm

        reproducing_individuals_amount = int(np.round(current_reproduction_rate * population_amount_before))

        # reproduction
        # the best individual is always reproduced, the rest are drawn randomly
        if verbose > 1:
            print(f"\r\tDoing reproduction...", end="")

        if reproducing_individuals_amount > 0:
            self.population.append(old_population_sorted[0])

            if reproducing_individuals_amount > 1:
                self.population.extend(
                    self.rng.choice(a=old_population_sorted[1:],
                                    size=reproducing_individuals_amount - 1,
                                    p=fitness_weights[1:] / np.sum(fitness_weights[1:]),
                                    replace=False)
                )

        if current_reproduction_rate < 1.:
            # crossover
            crossovers_to_do = int(len(old_population) * current_crossover_rate)
            tournament_size = int(self.settings.population_size * self.settings.tournament_percent_size)

            # make tournament size even (and not bigger)
            tournament_size -= tournament_size % 2
            tournament_size = max(2, tournament_size)

            crossover_start = time.time()

            if verbose > 1:
                print(f"\r                                           ", end="")
                print(f"\r\tDoing crossover...", end="")

            for i in range(crossovers_to_do):
                ph_e_attempts = max(self.settings.phenotype_exploration_attempts_during_crossover,
                                    1) if self.settings.fitness_log_is_phenotype_mapper else 1
                depth_attempts = max(self.settings.depth_attempts_during_crossover, 1)

                new_individual = None

                for _ in range(depth_attempts):
                    for _ in range(ph_e_attempts):
                        participants = self.rng.choice(a=len(old_population), size=tournament_size, replace=False)
                        participants_1 = participants[0:tournament_size // 2]
                        participants_2 = participants[tournament_size // 2:tournament_size]

                        assert len(participants_1) == len(participants_2)

                        best_participant_i_1 = np.argmin(
                            [fitness_values[participant] for participant in participants_1])
                        best_participant_i_2 = np.argmin(
                            [fitness_values[participant] for participant in participants_2])

                        new_individual = self.combine_individuals(individual_1=old_population[best_participant_i_1],
                                                                  individual_2=old_population[best_participant_i_2],
                                                                  verbose=verbose)

                        if repr(new_individual) not in self.fitness_log:
                            break

                    if new_individual.depth() <= self.settings.tree_transformation_max_depth:
                        break

                self.population.append(new_individual)

                if verbose > 1:
                    print(
                        f"\r\tDoing crossover... ETA {misc.timeformat_hhmmss(misc.timeleft(crossover_start, time.time(), i + 1, crossovers_to_do))}",
                        end="")

            # mutation
            if verbose > 1:
                print(f"\r                                           ", end="")
                print(f"\r\tDoing mutation...", end="\r")

            mutated_individuals_added = 0
            while len(self.population) < population_amount_before:
                individual_to_mutate = self.rng.choice(a=old_population_sorted, p=fitness_weights)

                new_individual = self.mutate_individual(
                    individual=individual_to_mutate,
                    inplace=False)

                self.population.append(new_individual)

                mutated_individuals_added += 1
        else:
            current_crossover_rate, crossovers_to_do, current_mutation_rate, mutated_individuals_added = 0., 0, 0., 0

        if verbose > 1:
            print(
                f"\t{current_reproduction_rate:.1%} ({reproducing_individuals_amount}) reproduction, {current_crossover_rate:.1%} ({crossovers_to_do}) crossover, {current_mutation_rate:.1%} ({mutated_individuals_added}) mutation")

        return result

    def run_genetic_algorithm(self, max_individuals_to_evaluate=-1, verbose=0):
        start = time.time()

        if verbose > 0:
            print("Running genetic algorithm...")

        individuals_evaluated_total = 0

        genalgo_log = None
        if self.settings.save_logs:
            genalgo_log = pd.DataFrame(columns=["Step", "Individual", "Fitness"])

        routine_output = None
        error_to_raise_later = None
        try:
            for step in range(1, self.settings.total_steps + 1):
                if 0 < max_individuals_to_evaluate < individuals_evaluated_total:
                    if verbose > 1:
                        print("Genetic algorithm evaluated maximum allowed number of individuals")
                    break

                if verbose > 1:
                    print(f"Step {step}")

                routine_output = self.do_genetic_routine_once(current_step=step, max_steps=self.settings.total_steps,
                                                              verbose=verbose)

                if self.settings.save_logs:
                    genalgo_log = pd.concat(objs=[genalgo_log,
                                                  pd.DataFrame({
                                                      "Step": step,
                                                      "Individual": routine_output.population_data["Individual"],
                                                      "Fitness": routine_output.population_data["Fitness"]
                                                  })
                                                  ],
                                            ignore_index=True)

                individuals_evaluated_total += routine_output.individuals_evaluated
        except KeyboardInterrupt as kb_interrupt:
            print("\nGenetic algorithm was manually interrupted")
            pass
        except Exception as error:
            print(f"\nGenetic algorithm stopped due to error \'{error}\'")
            error_to_raise_later = error
            pass

        did_at_least_one = routine_output is not None

        if verbose > 0:
            if did_at_least_one:
                print(
                    f"Done. Here is the best performing individual of the last step (with fitness {self.fitness_log[repr(routine_output.best_individual)]:.2f}):")
                print(routine_output.best_individual)
            else:
                print("There was no simulation output")

            print(f"Genetic simulation took {misc.timeformat(time.time() - start)}")

        if did_at_least_one and self.settings.save_logs:
            fitness_log_data = {"Individual": list(self.fitness_log.keys()),
                                "Fitness": list(self.fitness_log.values())}

            if type(self.fitness_log) == pht.PhenotypeMapper:
                fitness_log_data["Phenotype"] = [self.fitness_log.individual_to_phenotype[ind] for ind in
                                                 fitness_log_data["Individual"]]

            fitness_log_dataframe = pd.DataFrame(data=fitness_log_data)

            fitness_log_dataframe["string_length"] = fitness_log_dataframe.apply(lambda row: len(row["Individual"]),
                                                                                 axis=1)

            fitness_log_dataframe.sort_values(by=["Fitness", "string_length"], inplace=True)

            fitness_log_dataframe.drop(columns="string_length", inplace=True)

            foldername = datetime.datetime.now().strftime('%d %b %Y %H_%M_%S')

            if self.settings.fitness_is_random:
                foldername = f"{foldername} random fitness"

            filepath_fitnesslog_str = f"{foldername}/{FITNESSLOG_CSV_NAME}.csv"
            filepath_fitnesslog = Path(filepath_fitnesslog_str)
            filepath_fitnesslog.parent.mkdir(parents=True, exist_ok=True)

            try:
                fitness_log_dataframe.to_csv(path_or_buf=filepath_fitnesslog, index=False)

                if verbose > 0:
                    print("Fitness log saved successfully")
            except Exception as error:
                if verbose > 0:
                    print(f"Could not save fitness log ({error})")

            try:
                genalgo_log_path = f"{foldername}/{GENALGOLOG_CSV_NAME}.csv"

                genalgo_log.to_csv(path_or_buf=genalgo_log_path, index=False)

                if verbose > 0:
                    print("Genetic algorithm log saved successfully")
            except Exception as error:
                if verbose > 0:
                    print(f"Could not save genetic algorithm log ({error})")

            try:
                laststate_json_path = f"{foldername}/{LASTSTATE_JSON_NAME}.json"

                laststate_dict = dict()

                laststate_dict["rng_state"] = self.rng.bit_generator.state
                laststate_dict["population"] = [repr(individual) for individual in self.population]

                with open(laststate_json_path, 'w') as file:
                    json.dump(laststate_dict, file)

                if verbose > 0:
                    print("Last state saved successfully")
            except Exception as error:
                if verbose > 0:
                    print(f"Could not save last state ({error})")

        if verbose > 2:
            print("Fitness log:")
            print(misc.dictformat(self.fitness_log))

        if error_to_raise_later is not None:
            raise error_to_raise_later