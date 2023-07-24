import dfjss_misc as misc

RECIPES = {
    "heat": ["microwave", "gas_oven", "gas_burner", "electric_oven", "electric_burner"],
    "boil": ["gas_oven", "gas_burner", "electric_oven", "electric_burner"],
    "thaw": ["microwave"],
    "freeze": ["freezer"],
    "lick": ["thin_tongue", "wide_tongue"],
}

FAMILIES = list(RECIPES.keys())

# INTERVALS FOR NUMERIC VARIABLES

REQUIRES_INTEGERS = [
    "job_starting_number_of_operations",
    "simulation_number_of_starting_machines_over_essential",
    "simulation_number_of_starting_jobs",
    "machine_capacity"
]

NONNUMERIC_FEATURES = [
    "operation_family",
    "machine_recipe",
    "machine_capacity_scaling"
]

GENERATION_OPERATION_RANGES = {
    # WORK REQUIRED
    # Amount of "work" this operation requires.
    # The time to finish an operation is (operation_work_required / machine_nominal_work_power).
    # Units: "work" units (for example joules, seconds, etc)
    "operation_work_required": (10, 500),

    # WINDUP
    # When an operation is assigned to a machine, there is a
    # fixed amount of time before the operation starts.
    # The machine is *not free* during this process.
    # Units: seconds
    "operation_windup": (0, 120),

    # COOLDOWN
    # When a machine ends an operation, there is a
    # fixed amount of time before the job can continue.
    # The machine is *free* during this process.
    # Units: seconds
    "operation_cooldown": (0, 120),

    # START TIME
    # Absolute time at which the operation started.
    # Units: seconds
    "operation_start_time": -1
}

GENERATION_JOB_RANGES = {
    # NUMBER OF OPERATIONS
    # Number of operations the job starts with.
    # Units: potatoes
    "job_starting_number_of_operations": (2, 12),

    # ABSOLUTE/RELATIVE DEADLINE, INITIALIZATION TIME
    # Amount of time to complete the job. Related to i.e. net tardiness objective functions.
    # A late job does not affect the simulation.
    # Relative deadline changes in real time, while
    # absolute deadline and initialization time are set during job creation.
    # Units: seconds
    "job_absolute_deadline": -1,
    "job_relative_deadline": (60, 360),
    "job_initialization_time": -1,

    # REMAINING NUMBER OF OPERATIONS
    # Number of operations the job has left.
    # Changes in real time.
    # Units: potatoes
    "job_remaining_number_of_operations": -1,

    # REMAINING WORK TO COMPLETE
    # Amount of work left to complete this job.
    # Changes in real time.
    # Units: "work" units
    "job_remaining_work_to_complete": -1,
}

GENERATION_MACHINE_RANGES = {
    # NOMINAL WORK POWER
    # The rate at which this machine executes "work".
    # The time to finish an operation is (operation_work_required / machine_nominal_work_power).
    # Units: "work" units per second (for example watts, seconds/second, etc)
    "machine_nominal_work_power": (10, 500),

    # CAPACITY
    # How many operations can the machine do at once.
    #
    # Units: potatoes
    "machine_capacity": (1, 3),

    # CAPACITY SCALING
    # How the machine scales its work power, depending on capacity.
    # Accepted values:
    # -"constant": the machine will apply 100% of its nominal power to each operation
    # -"inverse": the machine will apply 100% / (no. of current operations) of its nominal power to each operation
    # -functions with two arguments, one containing the pair (machine and operation)'s features, and the other
    # being the number of operations currently ongoing
    # Units: percentage
    "machine_capacity_scaling": ["constant",
                                 "inverse",
                                 lambda pair_features, current_amount: (1. / current_amount) ** 0.5],

    # COOLDOWN
    # When a machine ends an operation, there is a
    # fixed amount of time before this machine can be used again.
    # Units: seconds
    "machine_cooldown": (0, 60),

    # FIXED/PER SECOND PROCESSING COST
    # Monetary cost of having the machine process, (per operation started/per second).
    # Units: money(/second)
    "machine_processing_cost_fixed": (0.01, 0.02),
    "machine_processing_cost_per_second": (0.005, 0.01),

    # FIXED/PER SECOND PROCESSING ENERGY
    # Energy used by having the machine process, (per operation started/per second).
    # Units: energy(/second)
    "machine_processing_energy_fixed": (1, 5),
    "machine_processing_energy_per_second": (10, 50),

    # START TIME
    # Absolute time at which the machine was created.
    # Units: seconds
    "machine_start_time": -1
}

GENERATION_WAREHOUSE_RANGES = {
    # UTILIZATION RATE
    # How many machines, in percentage, are currently being used.
    # Units: percent
    "warehouse_utilization_rate": 0.,
}

GENERATION_PAIR_RANGES = {
    # NUMBER OF ALTERNATIVE MACHINES
    # How many machines could also fulfill this operation, including the machine in the pair.
    # Units: potatoes
    "pair_number_of_alternative_machines": 0,

    # NUMBER OF ALTERNATIVE OPERATIONS
    # How many fulfillable operations could also be fulfilled by this machine, including the operation in the pair.
    # Units: potatoes
    "pair_number_of_alternative_operations": 0,

    # EXPECTED WORK POWER
    # Given the current situation, the work power that the machine will provide to the operation.
    # Units: "work" units per second (for example watts, seconds/second, etc)
    "pair_expected_work_power": -1,
    
    # NOMINAL PROCESSING TIME
    # The nominal time to finish an operation, equal to (operation_work_required / machine_nominal_work_power).
    # Units: seconds
    "pair_nominal_processing_time": 0.,

    # EXPECTED PROCESSING TIME
    # The nominal time to finish an operation, equal to (operation_work_required / machine_nominal_work_power).
    # Units: seconds
    "pair_expected_processing_time": 0.
    
}

GENERATION_SIMULATION_RANGES = {
    # NUMBER OF STARTING MACHINES OVER ESSENTIAL
    # Number of machines to generate at the start, beyond the "essential" ones (one for each family)
    # Units: potatoes
    "simulation_number_of_starting_machines_over_essential": 5,

    # NUMBER OF STARTING JOBS
    # Number of jobs to generate at the start
    # Units: potatoes
    "simulation_number_of_starting_jobs": 100,
}

# MANDATORY FEATURES
# Features are mandatory, like operation_family and machine_recipe...

MANDATORY_OPERATION_FEATURES = ["operation_family"]
MANDATORY_JOB_FEATURES = []
MANDATORY_MACHINE_FEATURES = ["machine_recipe"]
MANDATORY_WAREHOUSE_FEATURES = []
MANDATORY_SIMULATION_FEATURES = []
MANDATORY_PAIR_FEATURES = []

# ...and also whatever numeric feature was defined above

MANDATORY_OPERATION_FEATURES.extend(GENERATION_OPERATION_RANGES.keys())
MANDATORY_JOB_FEATURES.extend(GENERATION_JOB_RANGES.keys())
MANDATORY_MACHINE_FEATURES.extend(GENERATION_MACHINE_RANGES.keys())
MANDATORY_WAREHOUSE_FEATURES.extend(GENERATION_WAREHOUSE_RANGES.keys())
MANDATORY_SIMULATION_FEATURES.extend(GENERATION_SIMULATION_RANGES.keys())
MANDATORY_PAIR_FEATURES.extend(GENERATION_PAIR_RANGES.keys())

MANDATORY_FEATURES = misc.flatten([MANDATORY_OPERATION_FEATURES, MANDATORY_JOB_FEATURES, MANDATORY_MACHINE_FEATURES, MANDATORY_WAREHOUSE_FEATURES, MANDATORY_PAIR_FEATURES])

MANDATORY_NUMERIC_FEATURES = MANDATORY_FEATURES.copy()

for f in NONNUMERIC_FEATURES:
    if f in NONNUMERIC_FEATURES:
        MANDATORY_NUMERIC_FEATURES.remove(f)