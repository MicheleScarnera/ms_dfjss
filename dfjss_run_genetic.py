import dfjss_genetic as genetic


def custom_fitness_func(objectives):
    return objectives["max_completion_time"]


if __name__ == "__main__":
    gen_algo_settings = genetic.GeneticAlgorithmSettings()

    """
    gen_algo_settings.features = ["operation_work_required",
                                  "operation_windup",
                                  "operation_cooldown",
                                  "job_relative_deadline",
                                  "job_remaining_number_of_operations",
                                  "job_remaining_work_to_complete",
                                  "machine_capacity",
                                  "machine_cooldown",
                                  "machine_current_breakdown_rate",
                                  "machine_replacement_cooldown",
                                  "machine_processing_cost_fixed",
                                  "machine_processing_cost_per_second",
                                  "machine_processing_energy_fixed",
                                  "machine_processing_energy_per_second",
                                  "warehouse_utilization_rate",
                                  "pair_number_of_alternative_machines",
                                  "pair_number_of_alternative_operations",
                                  "pair_expected_work_power",
                                  "pair_expected_processing_time"]
    """

    gen_algo_settings.features = ["operation_windup",
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
                                  "pair_expected_processing_time"]

    #gen_algo_settings.warehouse_settings.generation_simulation_ranges["simulation_number_of_starting_jobs"] = 50
    #gen_algo_settings.warehouse_settings.generation_simulation_ranges["simulation_random_job_arrival_rate"] = 0
    #gen_algo_settings.warehouse_settings.generation_simulation_ranges["simulation_random_job_arrival_end_state_prevention_batch_size"] = 0

    #gen_algo_settings.fitness_func = custom_fitness_func
    gen_algo_settings.fitness_is_random = False

    gen_algo_settings.multiprocessing_processes = 5

    gen_algo_settings.population_size = 500
    gen_algo_settings.number_of_simulations_per_individual = 50
    gen_algo_settings.total_steps = 40

    gen_algo_settings.tree_generation_max_depth = 2
    gen_algo_settings.tree_generation_mode = "half_and_half"
    gen_algo_settings.tree_transformation_max_depth = 8

    gen_algo_settings.fitness_log_is_phenotype_mapper = True
    gen_algo_settings.phenotype_mapper_scenarios_amount = 100
    gen_algo_settings.phenotype_exploration_attempts_during_crossover = 3

    gen_algo_settings.depth_attempts_during_crossover = 3

    gen_algo_settings.reproduction_rate = 0.08
    gen_algo_settings.reproduction_rate_increment = 0.0032

    gen_algo_settings.crossover_rate = 0.9
    gen_algo_settings.crossover_rate_increment = 0.0184

    gen_algo_settings.mutation_rate = 0.02
    gen_algo_settings.mutation_rate_increment = 0.0184

    gen_algo_settings.tournament_percent_size = 0.05

    gen_algo = genetic.GeneticAlgorithm(settings=gen_algo_settings, rng_seed=123)

    #gen_algo.import_state("29 Aug 2023 17_59_26 random fitness")

    gen_algo.run_genetic_algorithm(max_individuals_to_evaluate=-1,
                                   verbose=2)
