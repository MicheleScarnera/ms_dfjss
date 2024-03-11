import dfjss_objects as dfjss
import dfjss_defaults as DEFAULTS
import dfjss_priorityfunction as pf
import dfjss_misc as misc

import time

rng_seed = 14825
max_steps = -1
verbose = 2

start = time.time()

pf_features = DEFAULTS.MANDATORY_FEATURES

branch = pf.representation_to_root_branch(
    representation="(((job_remaining_number_of_operations/(job_relative_deadline>{0.0}))/((pair_number_of_alternative_machines^warehouse_utilization_rate)*(pair_number_of_alternative_operations^({1.0}-warehouse_utilization_rate))))-{0.0})",
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

print("PRIORITY FUNCTION DECISION RULE")
print(repr(priorityfunc.root_branch))
print("")
print(priorityfunc.latex(parentheses=False))
print("")
print(prioritydecision_sim_out.summary())

print(f"Took {misc.timeformat(time.time() - start)}")
