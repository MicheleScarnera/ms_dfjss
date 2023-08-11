import dfjss_genetic as genetic

gen_algo_settings = genetic.GeneticAlgorithmSettings()

gen_algo_settings.features = ["operation_work_required",
                              "operation_windup",
                              "operation_cooldown",
                              "job_starting_number_of_operations",
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

gen_algo_settings.population_size = 300
gen_algo_settings.tournament_percent_size = 0.25
gen_algo_settings.total_steps = 15
gen_algo_settings.fitness_is_random = False
gen_algo_settings.number_of_simulations_per_individual = 1

gen_algo = genetic.GeneticAlgorithm(settings=gen_algo_settings, rng_seed=123)

gen_algo.run_genetic_algorithm(max_individuals_to_evaluate=-1,
                               verbose=2)
