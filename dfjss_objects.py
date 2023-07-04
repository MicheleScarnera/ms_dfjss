import numpy as np
from string import ascii_lowercase as alphabet

import dfjss_exceptions

DEFAULT_FAMILIES = ["heat", "boil", "thaw", "freeze", "lick", "kiss"]

DEFAULT_RECIPIES = {
    "heat": ["microwave", "gas_oven", "gas_burner", "electric_oven", "electric_burner"],
    "boil": ["gas_oven", "gas_burner", "electric_oven", "electric_burner"],
    "thaw": ["microwave"],
    "freeze": ["freezer"],
    "lick": ["thin_tongue", "wide_tongue"],
    "kiss": ["light_kiss", "french_kiss"],
}

MANDATORY_OPERATION_FEATURES = ["family"]
MANDATORY_JOB_FEATURES = []
MANDATORY_MACHINE_FEATURES = ["recipe"]


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


class DFJSSOperation:
    """

    """

    def __init__(self, features):
        self.features = features

        # check mandatory features
        check_mandatory_features(features_we_have=features, features_we_require=MANDATORY_OPERATION_FEATURES,
                                 name="Operation")


class DFJSSJob:
    """

    """

    def __init__(self, operations, features):
        self.operations = operations

        for operation in operations:
            if type(operation) != DFJSSOperation:
                raise dfjss_exceptions.JobWithBadOperationsError(
                    f"Some operations given to a job are not DFJSSOperation objects. Operations' types: {[type(op) for op in operations]}",
                    operations=operations
                )

        self.features = features

        # check mandatory features
        check_mandatory_features(features_we_have=features, features_we_require=MANDATORY_JOB_FEATURES,
                                 name="Job")


class DFJSSMachine:
    """

    """

    def __init__(self, features):
        self.features = features

        # check mandatory features
        check_mandatory_features(features_we_have=features, features_we_require=MANDATORY_MACHINE_FEATURES,
                                 name="Machine")
        