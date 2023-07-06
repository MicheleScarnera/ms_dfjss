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


class JobWithBadOperationsError(Exception):
    def __init__(self, message, operations):
        super().__init__(message)
        self.operations = operations

class WarehouseStuckError(Exception):
    def __init__(self, message, orphan_operations):
        super().__init__(message)
        self.orphan_operations = orphan_operations
