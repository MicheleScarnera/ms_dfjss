from typing import List, Any, Dict
import warnings

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

    def __init__(self, features, job=None):
        self.features = features

        # check mandatory features
        check_mandatory_features(features_we_have=features, features_we_require=DEFAULTS.MANDATORY_OPERATION_FEATURES,
                                 name="Operation")

        self.job = job


class Job:
    """

    """

    def __init__(self, operations, features):
        self.operations = operations

        for operation in operations:
            if type(operation) != Operation:
                raise dfjss_exceptions.JobWithBadOperationsError(
                    f"Some operations given to a job are not DFJSSOperation objects. Operations' types: {[type(op) for op in operations]}",
                    operations=operations
                )

            if operation.job is not None and operation.job != self:
                raise dfjss_exceptions.JobOperationsConflictError(
                    f"A new job is being given an operation that already belongs to another job",
                    old_job=operation.job,
                    new_job=self,
                    operation=operation
                )

            operation.job = self

        self.features = features

        # check mandatory features
        check_mandatory_features(features_we_have=features, features_we_require=DEFAULTS.MANDATORY_JOB_FEATURES,
                                 name="Job")


class Machine:
    """

    """

    def __init__(self, features):
        self.features = features

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
        raise NotImplementedError(
            "dfjss.BaseDecisionRule was used directly. Please make a class that inherits from BaseDecisionRule and overrides make_decision")


class RandomDecisionRule(BaseDecisionRule):
    def make_decision(self, warehouse):
        machines = warehouse.available_machines()
        jobs = warehouse.available_jobs()
        operations = warehouse.operations_from_available_jobs()

        compatible = np.array(
            [[machine_operation_compatible(machine, operation) for operation in operations] for machine in machines])

        if not np.any(compatible):
            return DecisionRuleOutput(success=False)
        else:
            m, o = warehouse.rng.choice(a=np.argwhere(compatible), size=1, axis=0)

            chosen_job = warehouse.job_of_operation(operations[o])

            return DecisionRuleOutput(success=True, machine=machines[m], job=chosen_job, operation=operations[o])


class DecisionRuleOutput:
    success: bool
    machine: Machine
    job: Job
    operation: Operation

    def __init__(self, success, machine=None, job=None, operation=None):
        self.success = success

        if success and (machine is None or job is None or operation is None):
            raise ValueError(
                f"DecisionRuleOutput has success=True but machine ({machine}), job ({job}) or operation ({operation}) are None")

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
        self.generation_warehouse_ranges = DEFAULTS.GENERATION_WAREHOUSE_RANGES


def generate_features(rng, ranges_dict):
    features = dict()
    # add other features
    for mandatory_feature, (v_low, v_high) in ranges_dict.items():
        if mandatory_feature in DEFAULTS.REQUIRES_INTEGERS:
            features[mandatory_feature] = rng.integers(low=int(v_low), high=int(v_high))
        else:
            features[mandatory_feature] = rng.uniform(low=v_low, high=v_high)

    return features


class BusyCouple:
    machine: Machine
    job: Job
    time_needed: float

    def __init__(self, machine, job, time_needed):
        self.machine = machine
        self.job = job
        self.time_needed = time_needed


class WaitingMachine:
    machine: Machine
    time_needed: float

    def __init__(self, machine, time_needed):
        self.machine = machine
        self.time_needed = time_needed


class WaitingJob:
    job: Job
    time_needed: float

    def __init__(self, job, time_needed):
        self.job = job
        self.time_needed = time_needed


class Warehouse:
    rng: np.random.Generator
    settings: WarehouseSettings
    machines: list[Machine]
    jobs: list[Job]
    busy_couples: list[BusyCouple]
    warehouse_features: dict[str, Any]

    def __init__(self, settings=None, rng_seed=None):
        self.rng = np.random.default_rng(seed=rng_seed)

        if settings is None:
            settings = WarehouseSettings()

        self.settings = settings

        self.machines = []
        self.jobs = []

        self.busy_couples = []
        self.waiting_machines = []
        self.waiting_jobs = []

        self.warehouse_features = generate_features(self.rng, self.settings.generation_warehouse_ranges)

        self.current_time = 0.

    def job_of_operation(self, operation):
        if operation.job is not None:
            return operation.job
        else:
            result = [job for job in self.jobs if job.operations[0] == operation][0]

            warnings.warn(
                dfjss_exceptions.WarehouseIncorrectlyOrphanOperationWarning(
                    message="Using Warehouse.job_of_operation(operation), while a job was found, operation.job is None. It has been assigned to the result",
                    assumed_job=result,
                    operation=operation
                )
            )

            operation.job = result
            return result

    def is_machine_busy(self, machine):
        for busy_couple in self.busy_couples:
            if busy_couple.machine == machine:
                return True

        for waiting_machine in self.waiting_machines:
            if waiting_machine.machine == machine:
                return True

        return False

    def is_job_busy(self, job):
        for busy_couple in self.busy_couples:
            if busy_couple.job == job:
                return True

        for waiting_job in self.waiting_machines:
            if waiting_job.job == job:
                return True

        return False

    def available_machines(self):
        return [machine for machine in self.machines if not self.is_machine_busy(machine)]

    def available_jobs(self):
        return [job for job in self.jobs if not self.is_job_busy(job)]

    def operations_from_available_jobs(self):
        return [job.operations[0] for job in self.jobs if not self.is_job_busy(job)]

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

    def create_operations(self, amount, job=None):
        result = []

        for i in range(amount):
            features = dict()

            # features
            # generate numeric features first
            features = generate_features(self.rng, self.settings.generation_operation_ranges)

            # add family
            features["operation_family"] = self.rng.choice(a=self.settings.families)

            new_operation = Operation(features=features, job=job)
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

    def assign_job_to_machine(self, job, machine):
        compatible = machine_operation_compatible(machine=machine, operation=job.operations[0])

        if not compatible:
            machine_families = [family for family, recipe in self.settings.families.items() if
                                recipe == machine.features["machine_recipe"]]
            raise dfjss_exceptions.WarehouseIncompatibleThingsError(
                f"Trying to assign a job to a machine, but they are incompatible (job's next operation's family is {job.operations[0].features['operation_family']} while the machine's is {machine_families})",
                job=job,
                machine=machine
            )

        job_already_busy = self.is_job_busy(job)
        machine_already_busy = self.is_machine_busy(machine)
        if job_already_busy or machine_already_busy:
            raise dfjss_exceptions.WarehouseAssigningBusyThingsError(
                f"Trying to assign a job to a machine, but at least one is already busy (Job: {'Busy' if job_already_busy else 'Not Busy'}, Machine: {'Busy' if machine_already_busy else 'Not Busy'})",
                job=job,
                machine=machine
            )

        operation = job.operations[0]
        time_needed = 0.

        # Operation's start time
        time_needed += operation.features["operation_start_time"]

        # Time needed to do operation
        time_needed += operation.features["operation_work_required"] / machine.features["machine_work_power"]

        new_couple = BusyCouple(machine=machine, job=job, time_needed=time_needed)

        self.busy_couples.append(new_couple)
        return new_couple

    def make_machine_wait(self, machine, time_needed):
        new_waiting_machine = WaitingMachine(machine=machine, time_needed=time_needed)

        self.waiting_machines.append(new_waiting_machine)

    def make_job_wait(self, job, time_needed):
        new_waiting_job = WaitingJob(job=job, time_needed=time_needed)

        self.waiting_jobs.append(new_waiting_job)

    def do_routine_once(self):
        # release things that can be released

        # operations that are finished
        for busy_couple in self.busy_couples.copy():
            if busy_couple.time_needed <= 0:
                self.busy_couples.remove(busy_couple)

                operation_done = busy_couple.job.operations.pop(0)

                # operation end times and machine cooldowns
                job_of_operation_done = self.job_of_operation(operation_done)
                job_wait_time = operation_done.features["operation_end_time"]
                self.make_job_wait(job=job_of_operation_done, time_needed=job_wait_time)

                machine_of_operation_done = busy_couple.machine
                machine_wait_time = machine_of_operation_done.features["machine_cooldown"]
                self.make_machine_wait(machine=machine_of_operation_done, time_needed=machine_wait_time)

        # jobs and machines that are waiting (operations' end times, machine cooldowns)
        for waiting_machine in self.waiting_machines.copy():
            if waiting_machine.time_needed <= 0:
                self.waiting_machines.remove(waiting_machine)

        for waiting_job in self.waiting_jobs.copy():
            if waiting_job.time_needed <= 0:
                self.waiting_jobs.remove(waiting_job)

        # assign operations to available machines according to the decision rule
        first_time = True
        decision_output = None
        while first_time or decision_output.success:
            first_time = False

            decision_output = self.settings.decision_rule.make_decision(warehouse=self)
            if decision_output.success:
                self.assign_job_to_machine(job=decision_output.job, machine=decision_output.machine)

        # progress time: time elapsed is the soonest amount of time when "something happens"

        times = [busy_couple.time_needed for busy_couple in self.busy_couples]
        times.extend([waiting_machine.time_needed for waiting_machine in self.waiting_machines])
        times.extend([waiting_job.time_needed for waiting_job in self.waiting_jobs])

        smallest_time = np.min(a=times)

        # elapse time

        for busy_couple in self.busy_couples:
            busy_couple.time_needed -= smallest_time

        for waiting_machine in self.waiting_machines:
            waiting_machine.time_needed -= smallest_time

        for waiting_job in self.waiting_jobs:
            waiting_job.time_needed -= smallest_time

        # return value: time elapsed?
        return smallest_time

    def simulate(self, duration=86400):
        # set all machines/jobs to not busy
        self.busy_couples = []

        # add 1 machine for each family, each machine has a random recipe from within the family
        # TODO: define creation settings for machines

        for family in self.settings.families:
            recipe = self.rng.choice(a=self.settings.recipes[family])

            self.add_machine(recipe=recipe)

        # plus some more

        for _ in range(self.warehouse_features["warehouse_number_of_starting_machines_over_essential"]):
            self.add_machine()

        # add jobs
        # TODO: define creation settings for jobs

        for _ in range(self.warehouse_features["warehouse_number_of_starting_jobs"]):
            self.add_job()

        # run routine...
        raise NotImplementedError()
