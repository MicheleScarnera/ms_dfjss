import dfjss_misc as misc

FAMILIES = ["heat", "boil", "thaw", "freeze", "lick", "kiss"]

RECIPES = {
    "heat": ["microwave", "gas_oven", "gas_burner", "electric_oven", "electric_burner"],
    "boil": ["gas_oven", "gas_burner", "electric_oven", "electric_burner"],
    "thaw": ["microwave"],
    "freeze": ["freezer"],
    "lick": ["thin_tongue", "wide_tongue"],
    "kiss": ["light_kiss", "french_kiss"],
}

# INTERVALS FOR NUMERIC VARIABLES

REQUIRES_INTEGERS = ["job_number_of_operations"]

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
    # Number of operations the job has.
    # Units: potatoes
    "job_number_of_operations": (2, 12),
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

# MANDATORY FEATURES
# Features are mandatory, like operation_family and machine_recipe...

MANDATORY_OPERATION_FEATURES = ["operation_family"]
MANDATORY_JOB_FEATURES = []
MANDATORY_MACHINE_FEATURES = ["machine_recipe"]

# ...and also whatever numeric feature was defined above

MANDATORY_OPERATION_FEATURES.extend(GENERATION_OPERATION_RANGES.keys())
MANDATORY_JOB_FEATURES.extend(GENERATION_JOB_RANGES.keys())
MANDATORY_MACHINE_FEATURES.extend(GENERATION_MACHINE_RANGES.keys())

