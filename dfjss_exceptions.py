# Priority functions

class ForbiddenCharacterError(Exception):
    pass


class BadSyntaxRepresentationError(Exception):
    pass


class TooManyIterationsRepresentationError(Exception):
    def __init__(self, message, crumbs):
        super().__init__(message)
        self.crumbs = crumbs


class UnexpectedLoopEndRepresentationError(Exception):
    def __init__(self, message, crumbs):
        super().__init__(message)
        self.crumbs = crumbs


class BadFinalCrumbRepresentationError(Exception):
    def __init__(self, message, crumb):
        super().__init__(message)
        self.crumb = crumb


# DFJSS Objects

class MissingMandatoryFeaturesError(Exception):
    def __init__(self, message, missing_features):
        super().__init__(message)
        self.missing_features = missing_features


class MandatoryFeatureUnknownTypeWarning(Warning):
    def __init__(self, message, value):
        super().__init__(message)
        self.value = value


class JobWithBadOperationsError(Exception):
    def __init__(self, message, operations):
        super().__init__(message)
        self.operations = operations


class JobOperationsConflictError(Exception):
    def __init__(self, message, new_job, old_job, operation):
        super().__init__(message)
        self.new_job = new_job
        self.old_job = old_job
        self.operation = operation


class MachineBadScalingFunctionError(Exception):
    def __init__(self, message, machine):
        super().__init__(message)
        self.machine = machine


class WarehouseIncompatibleThingsError(Exception):
    def __init__(self, message, job, machine):
        super().__init__(message)
        self.job = job
        self.machine = machine


class WarehouseAssigningBusyThingsError(Exception):
    def __init__(self, message, job, machine):
        super().__init__(message)
        self.job = job
        self.machine = machine


class WarehouseIncorrectlyOrphanOperationWarning(Warning):
    def __init__(self, message, assumed_job, operation):
        super().__init__(message)
        self.assumed_job = assumed_job
        self.operation = operation


class WarehouseStuckError(Exception):
    def __init__(self, message, orphan_operations):
        super().__init__(message)
        self.orphan_operations = orphan_operations
