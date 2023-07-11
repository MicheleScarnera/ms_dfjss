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
    "simulation_number_of_starting_jobs"
]

GENERATION_OPERATION_RANGES = {
    # WORK REQUIRED
    # Amount of "work" this operation requires.
    # The time to finish an operation is (operation_work_required / machine_work_power).
    # Units: "work" units (for example joules, seconds, etc)
    "operation_work_required": (10, 500),

    # START TIME
    # When an operation is assigned to a machine, there is a
    # fixed amount of time before the operation starts.
    # The machine is *not free* during this process.
    # Units: seconds
    "operation_start_time": (0, 120),

    # END TIME
    # When a machine ends an operation, there is a
    # fixed amount of time before the job can continue.
    # The machine is *free* during this process.
    # Units: seconds
    "operation_end_time": (0, 120)
}

GENERATION_JOB_RANGES = {
    # NUMBER OF OPERATIONS
    # Number of operations the job starts with.
    # Units: potatoes
    "job_starting_number_of_operations": (2, 12),

    # ABSOLUTE/RELATIVE DEADLINE
    # Amount of time to complete the job. Related to i.e. net tardiness objective functions.
    # A late job does not affect the simulation.
    # Relative deadline changes in real time, absolute deadline is set during job creation.
    # Units: seconds
    "job_absolute_deadline": -1,
    "job_relative_deadline": (600, 3600),

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
    # WORK POWER
    # The rate at which this machine executes "work".
    # The time to finish an operation is (operation_work_required / machine_work_power).
    # Units: "work" units per second (for example watts, seconds/second, etc)
    "machine_work_power": (10, 500),

    # COOLDOWN
    # When a machine ends an operation, there is a
    # fixed amount of time before this machine can be used again.
    # Units: seconds
    "machine_cooldown": (0, 60),
}

GENERATION_WAREHOUSE_RANGES = {
    # UTILIZATION RATE
    # How many machines, in percentage, are currently being used.
    # Units: percent
    "warehouse_utilization_rate": 0.,
}

GENERATION_PAIR_RANGES = {
    # NUMBER OF COMPATIBLE MACHINES
    # How many machines could also fulfill this operation, including the machine in the pair.
    # Units: potatoes
    "pair_number_of_compatible_machines": 0,

    # NUMBER OF COMPATIBLE OPERATIONS
    # How many fulfillable operations could also be fulfilled by this machine, including the operation in the pair.
    # Units: potatoes
    "pair_number_of_compatible_operations": 0,

    # PROCESSING TIME
    # The time to finish an operation, equal to (operation_work_required / machine_work_power).
    # Units: seconds
    "pair_processing_time": 0.
}

GENERATION_SIMULATION_RANGES = {
    # NUMBER OF STARTING MACHINES OVER ESSENTIAL
    # Number of machines to generate at the start, beyond the "essential" ones (one for each family)
    # Units: potatoes
    "simulation_number_of_starting_machines_over_essential": (1, 10),

    # NUMBER OF STARTING JOBS
    # Number of jobs to generate at the start
    # Units: potatoes
    "simulation_number_of_starting_jobs": (10, 25),
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
