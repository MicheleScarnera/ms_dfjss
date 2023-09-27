import dfjss_genetic as genetic
import dfjss_objects as dfjss
import dfjss_priorityfunction as pf


def custom_fitness_func(objectives):
    return objectives["max_completion_time"]


if __name__ == "__main__":
    verbose = 2

    gen_algo_settings = genetic.GeneticAlgorithmSettings()

    gen_algo_settings.features = ["operation_work_required",
                                  "operation_windup",
                                  "operation_cooldown",
                                  "job_relative_deadline",
                                  "job_time_alive",
                                  "job_remaining_number_of_operations",
                                  "job_remaining_work_to_complete",
                                  "job_earliness_penalty",
                                  "job_lateness_penalty",
                                  "job_delivery_relaxation",
                                  "machine_capacity",
                                  "machine_cooldown",
                                  "machine_current_breakdown_rate",
                                  "machine_replacement_cooldown",
                                  "warehouse_utilization_rate",
                                  "pair_number_of_alternative_machines",
                                  "pair_number_of_alternative_operations",
                                  "pair_expected_work_power",
                                  "pair_expected_processing_time"]

    gen_algo_settings.fitness_is_random = False

    gen_algo_settings.multiprocessing_processes = 5

    gen_algo_settings.population_size = 500
    gen_algo_settings.number_of_simulations_per_individual = 50
    gen_algo_settings.number_of_possible_seeds = 500
    gen_algo_settings.total_steps = 40

    gen_algo_settings.tree_generation_max_depth = 2
    gen_algo_settings.tree_generation_mode = "half_and_half"
    gen_algo_settings.tree_transformation_max_depth = 8

    gen_algo_settings.fitness_log_is_phenotype_mapper = False
    gen_algo_settings.phenotype_mapper_scenarios_amount = 16
    gen_algo_settings.phenotype_exploration_attempts_during_crossover = 3

    gen_algo_settings.depth_attempts_during_crossover = 3

    gen_algo_settings.reproduction_rate = 0.1
    #gen_algo_settings.reproduction_rate_increment = 0.0032

    gen_algo_settings.crossover_rate = 0.8
    #gen_algo_settings.crossover_rate_increment = 0.0184

    gen_algo_settings.mutation_rate = 0.1
    #gen_algo_settings.mutation_rate_increment = 0.0184

    gen_algo = genetic.GeneticAlgorithm(settings=gen_algo_settings, rng_seed=123)

    """
    baseline = "((job_time_alive/(job_relative_deadline>0))*job_remaining_number_of_operations)"

    for n in ["1.0"]:
        gen_algo.population.append(pf.representation_to_priority_function_tree(
            f"({baseline}-{n})",
            features=gen_algo.settings.features, operations=gen_algo.settings.operations))

    gen_algo_settings.population_size = len(gen_algo.population)
    gen_algo_settings.total_steps = 1
    """

    gen_algo.run_genetic_algorithm(max_individuals_to_evaluate=-1,
                                   verbose=verbose)
