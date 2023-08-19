from typing import List, Any, Dict
import warnings
from collections import Counter
import time

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


# DECISION RULES

class BaseDecisionRule:
    def make_decisions(self, warehouse):
        raise NotImplementedError(
            "dfjss.BaseDecisionRule was used directly. Please make a class that inherits from BaseDecisionRule and overrides make_decision")


class RandomDecisionRule(BaseDecisionRule):
    def make_decisions(self, warehouse):
        compatible_pairs = warehouse.compatible_pairs(include_busy=False)

        if len(compatible_pairs) <= 0:
            return DecisionRuleOutput(success=False)

        pairs = []
        while len(compatible_pairs) > 0:
            m, j = warehouse.rng.choice(a=compatible_pairs, size=1).reshape(2)
            pairs.append((m, j))

            compatible_pairs = [(machine, job) for machine, job in compatible_pairs if machine != m and job != j]

        return DecisionRuleOutput(success=True, pairs=pairs)


class DecisionRuleOutput:
    success: bool
    pairs: list[(Machine, Job)]

    def __init__(self, success, pairs=None):
        self.success = success

        if success and (pairs is None or len(pairs) <= 0):
            raise ValueError(
                f"DecisionRuleOutput has success=True but no pairs have been given")

        self.pairs = pairs

# WAREHOUSE


class WarehouseSettings:
    decision_rule: BaseDecisionRule

    def __init__(self):
        self.families = DEFAULTS.FAMILIES
        self.recipes = DEFAULTS.RECIPES

        self.decision_rule = RandomDecisionRule()

        self.generation_operation_ranges = DEFAULTS.GENERATION_OPERATION_RANGES
        self.generation_job_ranges = DEFAULTS.GENERATION_JOB_RANGES
        self.generation_machine_ranges = DEFAULTS.GENERATION_MACHINE_RANGES
        self.generation_warehouse_ranges = DEFAULTS.GENERATION_WAREHOUSE_RANGES
        self.generation_simulation_ranges = DEFAULTS.GENERATION_SIMULATION_RANGES

        self.generation_pair_ranges = DEFAULTS.GENERATION_PAIR_RANGES


def generate_features(rng, ranges_dict):
    features = dict()
    # add other features
    for mandatory_feature, value in ranges_dict.items():
        if type(value) is tuple and len(value) == 2 and misc.is_number(value[0]) and misc.is_number(value[1]):
            # uniform range
            (v_low, v_high) = value

            if mandatory_feature in DEFAULTS.REQUIRES_INTEGERS:
                features[mandatory_feature] = rng.integers(low=int(v_low), high=int(v_high))
            else:
                features[mandatory_feature] = rng.uniform(low=v_low, high=v_high)
        elif type(value) is list:
            # list of possible values
            features[mandatory_feature] = rng.choice(a=value)
        else:
            # constant
            features[mandatory_feature] = value

    return features


class ExpectedProcessingTimeNeededBusyCoupleOutput:
    def __init__(self, expected_time, scaling_factor):
        self.expected_time = expected_time
        self.scaling_factor = scaling_factor


class BusyCouple:
    machine: Machine
    job: Job
    nominal_processing_time_needed: float
    machine_windup: float

    def __init__(self, machine, job, nominal_processing_time_needed, machine_windup):
        self.machine = machine
        self.job = job
        self.nominal_processing_time_needed = nominal_processing_time_needed
        self.machine_windup = machine_windup

    def expected_processing_time_needed(self, warehouse):
        pair_features = warehouse.all_features_of_compatible_pair(machine=self.machine, job=self.job)

        scaling_factor = warehouse.scaling_factor_of_machine(self.machine,
                                                             directly_provided_features=pair_features)

        return ExpectedProcessingTimeNeededBusyCoupleOutput(
            self.nominal_processing_time_needed / scaling_factor,
            scaling_factor)


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


class WarehouseRoutineOutput:
    time_passed: float
    end_simulation: bool

    def __init__(self, time_passed, end_simulation=False):
        self.time_passed = time_passed
        self.end_simulation = end_simulation


class WarehouseSimulationOutput:
    success: bool
    simulation_time: float
    times_passed: list[float]
    job_times: list[float]
    job_relative_deadlines: list[float]
    workloads: list[float]
    machine_lifespans: list[float]

    costs: list[float]
    energies_used: list[float]

    def __init__(self, simulation_time=None,
                 times_passed=None,
                 job_times=None,
                 job_relative_deadlines=None,
                 workloads=None,
                 machine_lifespans=None,
                 costs=None,
                 energies_used=None):
        self.simulation_time = simulation_time

        self.times_passed = times_passed

        self.job_times = job_times
        self.job_relative_deadlines = job_relative_deadlines
        self.workloads = workloads

        self.machine_lifespans = machine_lifespans

        self.costs = costs
        self.energies_used = energies_used

    def get_objectives(self):
        result = dict()
        jt = np.array(self.job_times)
        jrd = np.array(self.job_relative_deadlines)
        wl = np.array(self.workloads)
        ml = np.array(self.machine_lifespans)
        c = np.array(self.costs)
        e = np.array(self.energies_used)

        result["max_completion_time"] = self.simulation_time

        result["mean_net_earliness"] = np.mean(jrd)
        result["mean_earliness"] = np.mean(jrd, where=jrd > 0.) if np.any(jrd > 0.) else 0.
        result["mean_tardiness"] = -np.mean(jrd, where=jrd < 0.) if np.any(jrd < 0.) else 0.
        result["max_tardiness"] = -np.min(jrd) if np.any(jrd < 0.) else 0.

        result["total_running_time"] = np.sum(ml)

        result["total_workload"] = np.sum(wl)
        result["max_workload"] = np.max(wl)

        result["total_idle_time"] = result["total_running_time"] - result["total_workload"]

        result["total_flow_time"] = np.sum(jt)
        result["mean_flow_time"] = np.mean(jt)

        result["total_operating_cost"] = np.sum(c)
        result["total_energy_consumption"] = np.sum(e)

        return result


    def summary(self):
        result = "SIMULATION OUTPUT\n"

        objectives = self.get_objectives()

        for key, value in objectives.items():
            result += f"\'{key}\': "

            if np.isnan(value):
                result += "NaN"
            else:
                if "time" in key or "earliness" in key or "tardiness" in key or "workload" in key:
                    result += f"{misc.timeformat(value)}"
                else:
                    result += f"{value:.2f}"

            result += "\n"

        return result


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

        self.simulation_features = generate_features(self.rng, self.settings.generation_simulation_ranges)

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

    def machine_operation_compatible(self, machine, operation):
        """

        :type machine: Machine
        :type operation: Operation
        """
        return machine.features["machine_recipe"] in self.settings.recipes[operation.features["operation_family"]]

    def compatible_pairs(self, include_busy=False):
        """
        Returns all (Machine, Job) pairs that are compatible. Can optionally include machines and jobs that are busy.

        :param include_busy: If True, includes machines and jobs that are busy.
        :return: list[(Machine, Job)]
        """
        machines = self.machines if include_busy else self.available_machines()
        jobs = self.jobs if include_busy else self.available_jobs()
        """
        result = misc.flatten(
            [
                [(machine, job)
                 for job in jobs if self.machine_operation_compatible(machine, job.operations[0])]
                for machine in machines
            ]
        )
        """

        result = []

        for machine in machines:
            for job in jobs:
                if self.machine_operation_compatible(machine, job.operations[0]):
                    result.append((machine, job))

        return result

    def scaling_factor_of_machine(self, machine, directly_provided_features=None):
        if directly_provided_features is not None:
            features = directly_provided_features
        else:
            features = machine.features

        scaling_func = features["machine_capacity_scaling"]
        n = max(len([None for couple in self.busy_couples if couple.machine == machine]), 1)

        if scaling_func == "constant":
            scaling_factor = 1.
        elif scaling_func == "inverse":
            scaling_factor = 1. / n
        elif callable(scaling_func):
            try:
                scaling_factor = scaling_func(features, n)
            except TypeError as type_error:
                raise dfjss_exceptions.MachineBadScalingFunctionError(
                    message="Value of machine_capacity_scaling, while being a function, does not take 2 arguments",
                    machine=machine
                )
        else:
            raise ValueError(
                "Value of machine_capacity_scaling is not a known string preset, nor a function that takes 2 arguments"
            )

        return scaling_factor

    def concurrent_operations_processing_under_machine(self, machine):
        return np.sum([busy_couple.machine == machine for busy_couple in self.busy_couples], dtype=int)

    def concurrent_cooldowns_of_machine(self, machine):
        return np.sum([waiting_machine.machine == machine for waiting_machine in self.waiting_machines], dtype=int)

    def slots_used_up_on_machine(self, machine):
        return self.concurrent_operations_processing_under_machine(machine) + self.concurrent_cooldowns_of_machine(machine)

    def is_machine_busy(self, machine):
        return self.slots_used_up_on_machine(machine) >= machine.features["machine_capacity"]

    def is_job_busy(self, job):
        for busy_couple in self.busy_couples:
            if busy_couple.job == job:
                return True

        for waiting_job in self.waiting_jobs:
            if waiting_job.job == job:
                return True

        return False

    def available_machines(self):
        return [machine for machine in self.machines if not self.is_machine_busy(machine)]

    def available_jobs(self):
        #return [job for job in self.jobs if not self.is_job_busy(job)]
        unavailable_jobs = set([busy_couple.job for busy_couple in self.busy_couples])
        unavailable_jobs.update([waiting_job.job for waiting_job in self.waiting_jobs])

        return [job for job in self.jobs if job not in unavailable_jobs]

    def operations_from_available_jobs(self):
        return [job.operations[0] for job in self.available_jobs()]

    def first_operations_from_all_jobs(self):
        return [job.operations[0] for job in self.jobs]

    def can_operation_be_done(self, operation, eventually=True):
        return np.any([self.machine_operation_compatible(machine, operation) for machine in self.machines if
                       (eventually or self.is_machine_busy(machine))])

    def add_machine(self, recipe=None):
        if recipe is None:
            recipe = self.rng.choice(a=misc.dict_flatten_values(self.settings.recipes))

        # features
        # generate numeric features first
        features = generate_features(self.rng, self.settings.generation_machine_ranges)

        # machine specific features
        features["machine_recipe"] = recipe

        features["machine_start_time"] = self.current_time

        new_machine = Machine(features=features)
        self.machines.append(new_machine)

        return new_machine

    def remove_machine(self, machine, simulation_output=None):
        """

        :type machine: Machine
        :type simulation_output: WarehouseSimulationOutput
        """
        if machine not in self.machines:
            raise ValueError("Could not find machine to be removed in the warehouse's machines")

        displaced_jobs = []

        # liberate any busy couple containing it
        for busy_couple in self.busy_couples.copy():
            if busy_couple.machine != machine:
                continue

            operation_done = busy_couple.job.operations[0]
            job_wait_time = operation_done.features["operation_cooldown"]
            self.make_job_wait(job=busy_couple.job, time_needed=job_wait_time)

            self.busy_couples.remove(busy_couple)

            displaced_jobs.append(busy_couple.job)

        self.machines.remove(machine)

        if simulation_output is not None:
            # machine lifespan
            simulation_output.machine_lifespans.append(self.current_time - machine.features["machine_start_time"])

        return displaced_jobs

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

        # job-specific adjustments to features
        features["job_absolute_deadline"] = self.current_time + features["job_relative_deadline"]
        features["job_initialization_time"] = self.current_time

        new_job = Job(operations=self.create_operations(amount=features["job_starting_number_of_operations"]),
                      features=features)

        self.jobs.append(new_job)
        return new_job

    def families_of_machine(self, machine, force_one_value=False):
        result = [family for family, recipes in self.settings.recipes.items() if
                  machine.features["machine_recipe"] in recipes]
        if force_one_value:
            if len(result) > 1:
                raise ValueError("Machine has more than one family, but force_one_value is True")

            return result[0]

        return result

    def assign_job_to_machine(self, job, machine,
                              simulation_output=None,
                              precomputed_available_jobs=None):
        """

        :type precomputed_available_machines: list[Machine]
        :type precomputed_available_jobs: list[Job]
        :type simulation_output: WarehouseSimulationOutput
        """
        compatible = self.machine_operation_compatible(machine=machine, operation=job.operations[0])

        if not compatible:
            machine_families = self.families_of_machine(machine)
            raise dfjss_exceptions.WarehouseIncompatibleThingsError(
                f"Trying to assign a job to a machine, but they are incompatible (job's next operation's family is {job.operations[0].features['operation_family']} while the machine's is {machine_families})",
                job=job,
                machine=machine
            )

        if precomputed_available_jobs is None:
            job_already_busy = self.is_job_busy(job)
        else:
            job_already_busy = job not in precomputed_available_jobs

        machine_already_busy = self.is_machine_busy(machine)

        if job_already_busy or machine_already_busy:
            raise dfjss_exceptions.WarehouseAssigningBusyThingsError(
                f"Trying to assign a job to a machine, but at least one is already busy (Job: {'Busy' if job_already_busy else 'Not Busy'}, Machine: {'Busy' if machine_already_busy else 'Not Busy'})",
                job=job,
                machine=machine
            )

        operation = job.operations[0]

        # Take note of operation's start time
        # If already positive, then treat it as "already started some other time, but somehow failed to complete"
        # TODO: Chance for machines to break?
        if operation.features["operation_start_time"] < 0.:
            operation.features["operation_start_time"] = self.current_time

        # Operation's windup
        windup = operation.features["operation_windup"]

        # Time needed to do operation
        processing_time = operation.features["operation_work_required"] / machine.features["machine_nominal_work_power"]

        new_couple = BusyCouple(machine=machine,
                                job=job,
                                machine_windup=windup,
                                nominal_processing_time_needed=processing_time)

        if simulation_output is not None:
            # monetary/energy costs
            simulation_output.costs.append(machine.features["machine_processing_cost_fixed"])
            simulation_output.energies_used.append(machine.features["machine_processing_energy_fixed"])

        if precomputed_available_jobs is not None:
            precomputed_available_jobs.remove(job)

        self.busy_couples.append(new_couple)
        return new_couple

    def make_machine_wait(self, machine, time_needed):
        new_waiting_machine = WaitingMachine(machine=machine, time_needed=time_needed)

        self.waiting_machines.append(new_waiting_machine)

    def make_job_wait(self, job, time_needed):
        new_waiting_job = WaitingJob(job=job, time_needed=time_needed)

        self.waiting_jobs.append(new_waiting_job)

    def all_features_of_compatible_pair(self, machine, job):
        """
        Returns a dictionary with all of the relevant features of a given machine and job, including pair-specific ones, prefixed with 'pair_'.
        :type machine: Machine
        :type job: Job
        :return: dict[str, Any]
        """

        if not self.machine_operation_compatible(machine, job.operations[0]):
            raise dfjss_exceptions.WarehouseIncompatibleThingsError(
                message="Trying to get features of a machine-job pair that is not compatible",
                job=job,
                machine=machine
            )

        operation = job.operations[0]

        result = self.warehouse_features | machine.features | job.features | operation.features

        result["pair_number_of_alternative_machines"] = len([
            m for m in self.available_machines() if self.machine_operation_compatible(m, operation)
        ])

        result["pair_number_of_alternative_operations"] = len([
            j for j in self.available_jobs() if self.machine_operation_compatible(machine, j.operations[0])
        ])

        result["pair_nominal_processing_time"] = operation.features["operation_work_required"] / machine.features[
            "machine_nominal_work_power"]

        # do this one as late as possible, as it is going to use the "live" result
        result["pair_expected_work_power"] = machine.features[
                                                 "machine_nominal_work_power"] * self.scaling_factor_of_machine(
            machine=machine, directly_provided_features=result)

        result["pair_expected_processing_time"] = operation.features["operation_work_required"] / result[
            "pair_expected_work_power"]

        # TODO: custom pair features

        return result

    def do_routine_once(self, simulation_output, decision_rule_override=None, verbose=0):
        """

        :type simulation_output: WarehouseSimulationOutput
        :type verbose: int
        """
        # TODO: (real) time complexity of simulation is linear in the number of machines, but x^1.5 in the number of jobs. fix it or something
        # RELEASE THINGS THAT CAN BE RELEASED

        # machine breakdown
        last_time_passed = 0 if simulation_output.times_passed is None or len(simulation_output.times_passed) <= 0 else simulation_output.times_passed[-1]
        for machine in self.machines:
            breakdown_rate = machine.features["machine_current_breakdown_rate"] * last_time_passed
            if breakdown_rate > 0. and self.rng.uniform() < (1. - np.exp(-breakdown_rate)):
                # break down machine
                displaced_jobs = self.remove_machine(machine=machine, simulation_output=simulation_output)

                recipe = machine.features["machine_recipe"]

                new_machine = self.add_machine(recipe=recipe)

                self.make_machine_wait(machine=new_machine, time_needed=new_machine.features["machine_replacement_cooldown"])

                if verbose > 1:
                    print(
                        f"\tA \'{machine.features['machine_recipe']}\' machine has broken down and has displaced {len(displaced_jobs)} job(s). The machine worked for {misc.timeformat(simulation_output.machine_lifespans[-1])}, its replacement will be operational in {misc.timeformat(new_machine.features['machine_replacement_cooldown'])}"
                    )

        # operations that are finished
        for busy_couple in self.busy_couples.copy():
            if busy_couple.nominal_processing_time_needed <= 0:
                self.busy_couples.remove(busy_couple)

                operation_done = busy_couple.job.operations.pop(0)

                if verbose > 1:
                    print(
                        f"\tA \'{operation_done.features['operation_family']}\' operation with \'{busy_couple.machine.features['machine_recipe']}\' machine has finished, job and machine are going into cooldown")

                # output's workload
                workload = self.current_time - operation_done.features["operation_start_time"]
                simulation_output.workloads.append(workload)

                # operation and machine cooldowns
                job_of_operation_done = busy_couple.job
                if len(job_of_operation_done.operations) > 0:
                    job_wait_time = operation_done.features["operation_cooldown"]
                    self.make_job_wait(job=job_of_operation_done, time_needed=job_wait_time)
                else:
                    # job finished
                    self.jobs.remove(job_of_operation_done)

                    simulation_output.job_times.append(
                        self.current_time - job_of_operation_done.features["job_initialization_time"]
                    )
                    simulation_output.job_relative_deadlines.append(
                        job_of_operation_done.features["job_relative_deadline"]
                    )

                    if verbose > 1:
                        print(
                            f"\tA job was completed (Net earliness: {misc.timeformat(job_of_operation_done.features['job_relative_deadline'])})")

                machine_of_operation_done = busy_couple.machine
                machine_wait_time = machine_of_operation_done.features["machine_cooldown"]

                # monetary/energy costs
                simulation_output.costs.append(workload * machine_of_operation_done.features["machine_processing_cost_per_second"])
                simulation_output.energies_used.append(workload * machine_of_operation_done.features["machine_processing_energy_per_second"])

                self.make_machine_wait(machine=machine_of_operation_done, time_needed=machine_wait_time)

        # jobs and machines that are waiting (operations' end times, machine cooldowns)
        for waiting_machine in self.waiting_machines.copy():
            if waiting_machine.time_needed <= 0:
                self.waiting_machines.remove(waiting_machine)

                if verbose > 1:
                    print(f"\tAvailable: \'{waiting_machine.machine.features['machine_recipe']}\' machine")

        for waiting_job in self.waiting_jobs.copy():
            if waiting_job.time_needed <= 0:
                self.waiting_jobs.remove(waiting_job)

                if verbose > 1:
                    print(
                        f"\tAvailable: Job with \'{waiting_job.job.operations[0].features['operation_family']}\' operation")

        # random job arrivals
        if len(self.jobs) > 0:
            jobarrival_rate = last_time_passed * self.simulation_features["simulation_random_job_arrival_rate"]
            if jobarrival_rate > 0.:
                new_jobs = self.rng.poisson(lam=jobarrival_rate, size=None)

                if new_jobs > 0:
                    for _ in range(new_jobs):
                        new_job = self.add_job()

                    if verbose > 1:
                        print(
                            f"\tJob arrivals: {new_jobs} jobs have been added"
                        )

        elif self.simulation_features["simulation_random_job_arrival_end_state_prevention_batch_size"] > 0 and self.simulation_features["simulation_random_job_arrival_rate"] > 0:
            to_spawn = self.simulation_features["simulation_random_job_arrival_end_state_prevention_batch_size"]

            for _ in range(to_spawn):
                new_job = self.add_job()
                self.make_job_wait(job=new_job,
                                   time_needed=self.rng.exponential(
                                       scale=self.simulation_features["simulation_random_job_arrival_end_state_prevention_average_waiting_time"],
                                       size=None))

            if verbose > 1:
                print(
                    f"\t{to_spawn} jobs have been added at once to prevent the simulation from ending abruptly"
                )


        # REAL-TIME FEATURES

        # utilization rate
        self.warehouse_features["warehouse_utilization_rate"] = len(self.busy_couples) / np.sum([machine.features["machine_capacity"] for machine in self.machines])

        # jobs' real time features
        for job in self.jobs:
            job.features["job_remaining_number_of_operations"] = len(job.operations)

            job.features["job_remaining_work_to_complete"] = np.sum(
                [op.features["operation_work_required"] for op in job.operations]
            )

            job.features["job_relative_deadline"] = job.features["job_absolute_deadline"] - self.current_time

        # machines' real time features
        for machine in self.machines:
            machine.features["machine_current_breakdown_rate"] = self.rng.uniform() * machine.features["machine_max_breakdown_rate"]

        # TODO: custom real-time features

        if verbose > 1:
            print(f"\tUtilization rate: {self.warehouse_features['warehouse_utilization_rate']:.1%}")

        # assign operations to available machines according to the decision rule
        first_time = True
        decision_output = None

        precomputed_available_jobs = self.available_jobs()

        while first_time or decision_output.success:
            first_time = False

            decision_rule = self.settings.decision_rule if decision_rule_override is None else decision_rule_override

            decision_output = decision_rule.make_decisions(warehouse=self)
            if decision_output.success:
                for machine, job in decision_output.pairs:
                    self.assign_job_to_machine(job=job,
                                               machine=machine,
                                               simulation_output=simulation_output,
                                               precomputed_available_jobs=precomputed_available_jobs)

                    if verbose > 1:
                        print(
                            f"\tAssigned \'{machine.features['machine_recipe']}\' machine ({self.slots_used_up_on_machine(machine)}/{machine.features['machine_capacity']} capacity, {'custom' if callable(machine.features['machine_capacity_scaling']) else machine.features['machine_capacity_scaling']} scaling) to a \'{job.operations[0].features['operation_family']}\' operation")

        # CHECK END STATES

        # if there are no more jobs, end simulation

        if len(self.jobs) == 0:
            return WarehouseRoutineOutput(time_passed=0, end_simulation=True)

        # if there are no machines to do some operations, throw an error
        orphan_operations = [operation
                             for operation in self.first_operations_from_all_jobs()
                             if not self.can_operation_be_done(operation, eventually=True)]
        if len(orphan_operations) > 0:
            raise dfjss_exceptions.WarehouseStuckError(
                message=f"""There are operations that need to be done, but there are no machines to do so
\"Orphan\" Operations\' families: {np.unique([operation.features["operation_family"] for operation in orphan_operations])}""",
                orphan_operations=orphan_operations
            )

        # progress time: time elapsed is the soonest amount of time when "something happens"
        EPTs = [busy_couple.expected_processing_time_needed(warehouse=self) for busy_couple in self.busy_couples]

        times = [busy_couple.machine_windup if busy_couple.machine_windup > 0. else ept.expected_time
                 for busy_couple, ept in zip(self.busy_couples, EPTs)]
        times.extend([waiting_machine.time_needed for waiting_machine in self.waiting_machines])
        times.extend([waiting_job.time_needed for waiting_job in self.waiting_jobs])

        smallest_time = np.min(a=times)

        # ELAPSE TIME

        for busy_couple, ept in zip(self.busy_couples, EPTs):
            if busy_couple.machine_windup > 0.:
                busy_couple.machine_windup = min(busy_couple.machine_windup - smallest_time, 0.)
            else:
                busy_couple.nominal_processing_time_needed -= smallest_time * ept.scaling_factor

        for waiting_machine in self.waiting_machines:
            waiting_machine.time_needed -= smallest_time

        for waiting_job in self.waiting_jobs:
            waiting_job.time_needed -= smallest_time

        return WarehouseRoutineOutput(time_passed=smallest_time, end_simulation=False)

    def simulate(self, max_routine_steps=-1, decision_rule_override=None, verbose=0):
        if verbose > 0:
            print("Simulating warehouse...")

        realtime_start = time.time()

        # result
        simulation_output = WarehouseSimulationOutput(
            simulation_time=0,
            times_passed=[],
            job_times=[],
            job_relative_deadlines=[],
            workloads=[],
            machine_lifespans=[],
            costs=[],
            energies_used=[]

        )

        # set all machines/jobs to not busy and not waiting
        self.waiting_machines = []
        self.waiting_jobs = []
        self.busy_couples = []

        # store starting time
        starting_time = self.current_time

        # max simulation time
        max_simulation_time = self.simulation_features["simulation_max_simulation_time"]

        # add 1 machine for each family, each machine has a random recipe from within the family
        # TODO: define creation settings for machines

        for family in self.settings.families:
            recipe = self.rng.choice(a=self.settings.recipes[family])

            self.add_machine(recipe=recipe)

        # plus some more

        for _ in range(self.simulation_features["simulation_number_of_starting_machines_over_essential"]):
            self.add_machine()

        if verbose > 1:
            recipe_counter = Counter([machine.features["machine_recipe"] for machine in self.machines])
            print(f"Number of machines by recipe: {recipe_counter}")

        # add jobs
        # TODO: define creation settings for jobs

        for _ in range(self.simulation_features["simulation_number_of_starting_jobs"]):
            self.add_job()

        if verbose > 1:
            print(f"Number of jobs: {len(self.jobs)}")

        end_reason = "UNKNOWN"
        routine_step = 1
        # run routine...
        while True:
            if verbose > 1:
                print(f"Routine step {routine_step} - Time: {misc.timeformat(self.current_time)} - Jobs to do: {len(self.jobs)}")

            routine_result = self.do_routine_once(simulation_output=simulation_output, decision_rule_override=decision_rule_override, verbose=verbose)

            if routine_result.end_simulation:
                end_reason = "the routine ending it"
                break
            else:
                self.current_time += routine_result.time_passed

                simulation_output.times_passed.append(routine_result.time_passed)

                if verbose > 1:
                    print(f"Routine step {routine_step} ended - Time elapsed: {routine_result.time_passed:.2f}s",
                          end="\n\n")

            if max_simulation_time > 0 and self.current_time > (starting_time + max_simulation_time):
                end_reason = "reaching the maximum simulation time"
                break

            if 0 < max_routine_steps <= routine_step:
                end_reason = "reaching the maximum routine steps"
                break

            routine_step += 1

        # SIMULATION ENDS

        simulation_output.simulation_time = self.current_time - starting_time

        # machine lifespan
        for machine in self.machines:
            simulation_output.machine_lifespans.append(self.current_time - machine.features["machine_start_time"])

        if verbose > 0:
            print(f"Simulation ended in {misc.timeformat(simulation_output.simulation_time)} ({misc.timeformat(time.time() - realtime_start)} real time) due to {end_reason}")

        return simulation_output
