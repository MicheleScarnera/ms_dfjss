from typing import List, Any, Dict

import numpy as np

import dfjss_exceptions
import dfjss_defaults as DEFAULTS
import dfjss_misc as misc


def check_mandatory_features(features_we_have, features_we_require, name):
    mandatory_features_to_go = set(features_we_require.copy())
    for key, value in features_we_have.items():
        if key in mandatory_features_to_go:
            mandatory_features_to_go.remove(key)

    if len(mandatory_features_to_go) > 0:
        raise dfjss_exceptions.MissingMandatoryFeaturesError(
            f"{name} does not have all mandatory features. Missing features: {list(mandatory_features_to_go)}",
            missing_features=mandatory_features_to_go
        )


class Operation:
    """

    """

    def __init__(self, features):
        self.features = features

        # check mandatory features
        check_mandatory_features(features_we_have=features, features_we_require=DEFAULTS.MANDATORY_OPERATION_FEATURES,
                                 name="Operation")


class Job:
    """

    """

    def __init__(self, operations, features, busy=False):
        self.operations = operations

        self.busy = busy

        for operation in operations:
            if type(operation) != Operation:
                raise dfjss_exceptions.JobWithBadOperationsError(
                    f"Some operations given to a job are not DFJSSOperation objects. Operations' types: {[type(op) for op in operations]}",
                    operations=operations
                )

        self.features = features

        # check mandatory features
        check_mandatory_features(features_we_have=features, features_we_require=DEFAULTS.MANDATORY_JOB_FEATURES,
                                 name="Job")


class Machine:
    """

    """

    def __init__(self, features, busy=False):
        self.features = features

        self.busy = busy

        # check mandatory features
        check_mandatory_features(features_we_have=features, features_we_require=DEFAULTS.MANDATORY_MACHINE_FEATURES,
                                 name="Machine")

def machine_operation_compatible(machine, operation):
    """

    :type machine: Machine
    :type operation: Operation
    """
    return machine.features["machine_recipe"] in operation.features["operation_family"]


# DECISION RULES

class BaseDecisionRule:
    def make_decision(self, warehouse):
        raise NotImplementedError("dfjss.BaseDecisionRule was used directly. Please make a class that inherits from BaseDecisionRule and overrides make_decision")


class RandomDecisionRule(BaseDecisionRule):
    def make_decision(self, warehouse):
        machines = warehouse.available_machines()
        operations = warehouse.operations_from_available_jobs()

        compatible = np.array([[machine_operation_compatible(machine, operation) for operation in operations] for machine in machines])

        if not np.any(compatible):
            return DecisionRuleOutput(success=False)
        else:
            m, o = warehouse.rng.choice(a=np.argwhere(compatible), size=1, axis=0)

            return DecisionRuleOutput(success=True, machine=machines[m], operation=operations[o])


class DecisionRuleOutput:
    success: bool
    machine: Machine
    operation: Operation

    def __init__(self, success, machine=None, operation=None):
        self.success = success

        if success and (machine is None or operation is None):
            raise ValueError(f"DecisionRuleOutput has success=True but machine ({machine}) or operation ({operation}) are None")
        
        self.machine = machine
        self.operation = operation

# WAREHOUSE


class WarehouseSettings:
    def __init__(self):
        self.families = DEFAULTS.FAMILIES
        self.recipes = DEFAULTS.RECIPES

        self.decision_rule = RandomDecisionRule()

        self.generation_operation_ranges = DEFAULTS.GENERATION_OPERATION_RANGES
        self.generation_job_ranges = DEFAULTS.GENERATION_JOB_RANGES
        self.generation_machine_ranges = DEFAULTS.GENERATION_MACHINE_RANGES


def generate_features(rng, ranges_dict):
    features = dict()
    # add other features
    for mandatory_feature, (v_low, v_high) in ranges_dict.items():
        if mandatory_feature in DEFAULTS.REQUIRES_INTEGERS:
            features[mandatory_feature] = rng.integers(low=int(v_low), high=int(v_high))
        else:
            features[mandatory_feature] = rng.uniform(low=v_low, high=v_high)

    return features


class Warehouse:
    rng: np.random.Generator
    settings: WarehouseSettings
    machines: list[Machine]
    jobs: list[Job]
    warehouse_features: dict[str, Any]

    def __init__(self, settings=None, rng_seed=None):
        self.rng = np.random.default_rng(seed=rng_seed)

        if settings is None:
            settings = WarehouseSettings()

        self.settings = settings

        self.machines = []

        self.jobs = []

        self.warehouse_features = dict()

    def available_machines(self):
        return [machine for machine in self.machines if not machine.busy]

    def available_jobs(self):
        return [job for job in self.jobs if not job.busy]

    def operations_from_available_jobs(self):
        return [job.operations[0] for job in self.jobs if not job.busy]

    def add_machine(self, recipe=None):
        if recipe is None:
            recipe = self.rng.choice(a=misc.dict_flatten_values(self.settings.recipes))

        # features
        # generate numeric features first
        features = generate_features(self.rng, self.settings.generation_machine_ranges)

        # add recipe
        features["machine_recipe"] = recipe

        new_machine = Machine(features=features)
        self.machines.append(new_machine)

        return new_machine

    def create_operations(self, amount):
        result = []

        for i in range(amount):
            features = dict()

            # features
            # generate numeric features first
            features = generate_features(self.rng, self.settings.generation_operation_ranges)

            # add family
            features["operation_family"] = self.rng.choice(a=self.settings.families)

            new_operation = Operation(features=features)
            result.append(new_operation)

        return result

    def add_job(self):
        # features
        # generate numeric features first
        features = generate_features(self.rng, self.settings.generation_job_ranges)

        new_job = Job(operations=self.create_operations(amount=features["job_number_of_operations"]),
                            features=features)

        self.jobs.append(new_job)
        return new_job

    #def simulate(self, duration=120):