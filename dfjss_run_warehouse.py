import dfjss_objects as dfjss
import dfjss_defaults as DEFAULTS
import dfjss_priorityfunction as pf
import dfjss_misc as misc
import dfjss_phenotype as pht

import time

rng_seed = 14825
max_steps = -1
verbose = 0

start_beginning = time.time()

make_junk = True
if make_junk:
    if True:
        fitness_log = dict()

        precomputed_scenarios = None
        for seed in range(0, 1500):
            fitness_log[seed] = pht.PhenotypeMapper(scenarios_seed=rng_seed,
                                                    reference_scenarios_amount=25,
                                                    precomputed_scenarios=precomputed_scenarios)

            if precomputed_scenarios is None:
                precomputed_scenarios = fitness_log[seed].scenarios

        print(len(fitness_log.keys()))

        del precomputed_scenarios

        for i in list(fitness_log.keys()):
            del fitness_log[i]

        del fitness_log

    if False:
        w = dict()

        for i in range(1500):
            w[i] = dfjss.Warehouse()

start_warehouse = time.time()

# random decision

#randomdecision_warehouse = dfjss.Warehouse(rng_seed=rng_seed)

#randomdecision_sim_out = randomdecision_warehouse.simulate(max_routine_steps=max_steps, verbose=verbose)

# priority function decision

pf_features = DEFAULTS.MANDATORY_FEATURES

#print(pf_features)

# "((job_remaining_number_of_operations/(job_relative_deadline>0))/(pair_number_of_compatible_machines*pair_number_of_compatible_operations))"
# "(((job_remaining_number_of_operations/(job_relative_deadline>0))/((pair_number_of_alternative_machines^warehouse_utilization_rate)*(pair_number_of_alternative_operations^(1-warehouse_utilization_rate))))-2.0)"
branch = pf.representation_to_root_branch(
    representation='((job_relative_deadline*-4.5)<(-5.0<pair_number_of_alternative_machines))',
    features=pf_features)

priorityfunc = pf.PriorityFunctionTree(
    root_branch=branch,
    features=pf_features
)

prioritydecision_settings = dfjss.WarehouseSettings()

prioritydecision_settings.decision_rule = pf.PriorityFunctionTreeDecisionRule(
    priority_function_tree=priorityfunc
)

prioritydecision_warehouse = dfjss.Warehouse(settings=prioritydecision_settings, rng_seed=rng_seed)

prioritydecision_sim_out = prioritydecision_warehouse.simulate(max_routine_steps=max_steps, verbose=verbose)

#print("RANDOM DECISION RULE")
#print(randomdecision_sim_out.summary())

print("PRIORITY FUNCTION DECISION RULE")
print(repr(priorityfunc.root_branch))
print("")
print(priorityfunc.latex(parentheses=False))
print("")
print(prioritydecision_sim_out.summary())

print(f"Beginning: {misc.timeformat(start_warehouse - start_beginning)}")
print(f"Warehouse: {misc.timeformat(time.time() - start_warehouse)}")
print(f"Total: {misc.timeformat(time.time() - start_beginning)}")