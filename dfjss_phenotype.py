import numpy as np

import dfjss_objects as dfjss
import dfjss_priorityfunction as pf


class PhenotypeMapper:
    def __init__(self, reference_rule=None, reference_scenarios_amount=16, scenarios_seed=None, precomputed_scenarios=None):
        self.reference_scenarios_amount = reference_scenarios_amount
        self.dud_individual_std_threshold = 0.01

        if precomputed_scenarios is None:
            # go the extra mile to generate credible scenarios: generate them straight from a warehouse
            # it shouldn't be too slow since only one routine is ran

            self.warehouse_settings = dfjss.WarehouseSettings()

            self.warehouse_settings.generation_simulation_ranges["simulation_number_of_starting_jobs"] = reference_scenarios_amount

            self.warehouse_settings.generation_simulation_ranges["simulation_number_of_starting_machines_over_essential"] = len(self.warehouse_settings.recipes) * 2

            if reference_rule is not None:
                self.warehouse_settings.decision_rule = reference_rule

            self._warehouse = dfjss.Warehouse(settings=self.warehouse_settings, rng_seed=scenarios_seed)

            self._warehouse.simulate(max_routine_steps=1, verbose=0)

            compatible_pairs = self._warehouse.compatible_pairs(include_busy=True)

            self.scenarios = [self._warehouse.all_features_of_compatible_pair(machine=machine, job=job)
                              for machine, job in compatible_pairs]

            # if there are more than 'reference_scenarios_amount' scenarios, remove at random
            if len(self.scenarios) > reference_scenarios_amount:
                rng = np.random.default_rng(seed=scenarios_seed)
                self.scenarios = list(rng.choice(a=self.scenarios, size=reference_scenarios_amount, replace=False))
            elif len(self.scenarios) < reference_scenarios_amount:
                raise Exception(f"Number of scenarios in phenotype mapper is unexpectedly below specified amount {reference_scenarios_amount}")

            if reference_rule is None:
                rng = np.random.default_rng(seed=scenarios_seed)
                order = rng.permutation(x=reference_scenarios_amount)
            elif type(reference_rule) == pf.PriorityFunctionTreeDecisionRule:
                order = np.argsort(
                    [reference_rule.priority_function_tree.run(features=scenario) for scenario in scenarios_seed])
            else:
                raise ValueError(f"Reference rule of phenotype mapper is unexpected (of type {type(reference_rule)})")

            self.scenarios = [self.scenarios[i_sorted] for i_sorted in order]
        else:
            self.warehouse_settings = None
            self._warehouse = None

            self.scenarios = precomputed_scenarios

        self.features = list(self.scenarios[0].keys())

        # set up the two dictionaries
        self.individual_to_phenotype = dict()
        self.phenotype_to_fitness = dict()

    def get_phenotype_of_individual(self, individual_represenation):
        if type(individual_represenation) != str:
            raise ValueError("PhenotypeMapper only allows individuals in string form")

        try:
            individual = pf.representation_to_priority_function_tree(
                representation=individual_represenation,
                features=self.features)
        except Exception as error:
            raise ValueError(f"Could not recognize individual representation as a PriorityFunctionTree. The following exception was raised:\n{error}")

        priority_values = [individual.run(features=scenario) for scenario in self.scenarios]

        del individual

        if np.std(priority_values) > self.dud_individual_std_threshold:
            return tuple(np.argsort(priority_values))
        else:
            return tuple([0] * self.reference_scenarios_amount)

    def add_individual(self, individual_represenation, fitness):
        phenotype = self.get_phenotype_of_individual(individual_represenation)

        self.individual_to_phenotype[individual_represenation] = phenotype
        self.phenotype_to_fitness[phenotype] = fitness

    def __contains__(self, individual_representation):
        if type(individual_representation) != str:
            raise ValueError(f"PhenotypeMapper only allows individuals in string form for the \'contains\' operation  ({type(individual_representation)} was supplied)")

        if len(self.individual_to_phenotype.keys()) <= 0:
            return False

        phenotype = self.get_phenotype_of_individual(individual_representation)

        if phenotype in self.phenotype_to_fitness:
            # if we already have the fitness for this phenotype, add this individual right away and return true
            self.individual_to_phenotype[individual_representation] = phenotype
            return True

        # if we don't, return false
        return False

    def __getitem__(self, individual_representation):
        if type(individual_representation) != str:
            raise ValueError("PhenotypeMapper only allows individuals in string form for the \'get_item\' operation")

        phenotype = self.get_phenotype_of_individual(individual_representation)

        if phenotype in self.phenotype_to_fitness:
            # if we already have the fitness for this phenotype, add this individual right away
            self.individual_to_phenotype[individual_representation] = phenotype
            return self.phenotype_to_fitness[phenotype]

        raise KeyError(f"The individual is not in the PhenotypeMapper")

    def __setitem__(self, individual_representation, fitness):
        self.add_individual(individual_represenation=individual_representation, fitness=fitness)

    def __iter__(self):
        return iter([(individual_repr, self.phenotype_to_fitness[self.individual_to_phenotype[individual_repr]]) for individual_repr in self.individual_to_phenotype.keys()])

    def keys(self):
        return self.individual_to_phenotype.keys()

    def values(self):
        return [self.phenotype_to_fitness[self.individual_to_phenotype[individual_repr]] for individual_repr in self.keys()]

    def get(self, key, fallback_value=None):
        return self[key] if key in self else fallback_value

    def phenotype_amount(self):
        return len(self.phenotype_to_fitness.keys())
