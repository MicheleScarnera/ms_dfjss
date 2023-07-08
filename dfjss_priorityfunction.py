import numpy as np
from string import ascii_lowercase as alphabet
import numbers

import dfjss_exceptions
import dfjss_objects as dfjss

DEFAULT_FEATURES = alphabet

DEFAULT_OPERATIONS = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y if y != 0 else x,
    "<": lambda x, y: min(x, y),
    ">": lambda x, y: max(x, y),
}

FORBIDDEN_CHARACTERS = ['(', ')']


class PriorityFunctionTree:
    """
    This object is used to construct mathematical expressions, given a set of
    features and operations that it can have. The expression has a tree-like
    structure, resembling the tree-like priority functions that are used in
    Job Shop Scheduling (JSS) problems.
    The expression can be evaluated when passing the values of the features.
    """

    def __init__(self, root_branch, features=None, operations=None):
        if features is None:
            features = DEFAULT_FEATURES

        if operations is None:
            operations = DEFAULT_OPERATIONS

        assert_features_and_operations_validity(features, operations)

        self.features = features
        self.operations = operations
        self.root_branch = root_branch

    def __repr__(self):
        result = repr(self.root_branch)

        if self.features is not DEFAULT_FEATURES:
            result += f" (Custom Features: {self.features})"

        if self.operations is not DEFAULT_OPERATIONS:
            result += f" (Custom Operations: {self.operations})"

        return result

    def count(self):
        return self.root_branch.count()

    def depth(self):
        return self.root_branch.depth()

    def run(self, features):
        return self.root_branch.run(self.operations, features)


class PriorityFunctionBranch:
    """
    This object is a tree branch which takes a left feature, an operation, and a
    right feature. The operation is a string, while features can either be strings
    or other PriorityFunctionBranch objects. When paired with a PriorityFunctionTree
    object, which passes values for the features and the meaning of operations,
    they can be used to construct mathematical expressions. The structure resembles
    the tree-like priority functions that are used in Job Shop Scheduling (JSS) problems.
    """

    def __init__(self, left_feature, operation_character, right_feature):
        self.left_feature = left_feature
        self.right_feature = right_feature
        self.operation_character = operation_character

    def __repr__(self):
        if isinstance(self.left_feature, PriorityFunctionBranch):
            left = repr(self.left_feature)
        else:
            left = self.left_feature

        if isinstance(self.right_feature, PriorityFunctionBranch):
            right = repr(self.right_feature)
        else:
            right = self.right_feature

        return f"({left}{self.operation_character}{right})"

    def count(self):
        # Counts the number of sub-branches (including this one).
        tally = 1

        if isinstance(self.left_feature, PriorityFunctionBranch):
            tally += self.left_feature.count()

        if isinstance(self.right_feature, PriorityFunctionBranch):
            tally += self.right_feature.count()

        return tally

    def depth(self):
        # Counts the depth of sub-branches (a branch with no further branches has depth 1).
        if isinstance(self.left_feature, PriorityFunctionBranch):
            left_depth = self.left_feature.depth()
        else:
            left_depth = 0

        if isinstance(self.right_feature, PriorityFunctionBranch):
            right_depth = self.right_feature.depth()
        else:
            right_depth = 0

        return 1 + max(left_depth, right_depth)

    def run(self, operations, features_values):
        if isinstance(self.left_feature, PriorityFunctionBranch):
            left = self.left_feature.run(operations, features_values)
        elif isinstance(self.left_feature, numbers.Number):
            left = self.left_feature
        else:
            left = features_values[self.left_feature]

        if isinstance(self.right_feature, PriorityFunctionBranch):
            right = self.right_feature.run(operations, features_values)
        elif isinstance(self.right_feature, numbers.Number):
            right = self.right_feature
        else:
            right = features_values[self.right_feature]

        return operations[self.operation_character](left, right)


def assert_features_and_operations_validity(features, operations):
    for c in FORBIDDEN_CHARACTERS:
        if c in features:
            raise dfjss_exceptions.ForbiddenCharacterError(f"Forbidden character \'{c}\' found in features")

        if c in operations.keys():
            raise dfjss_exceptions.ForbiddenCharacterError(f"Forbidden character \'{c}\' found in operations")


def representation_to_priority_function_tree(representation, features=None, operations=None, max_iter=500, verbose=0):
    return PriorityFunctionTree(
        root_branch=representation_to_root_branch(representation=representation,
                                                  features=features,
                                                  operations=operations,
                                                  max_iter=max_iter,
                                                  verbose=verbose),
        features=features,
        operations=operations,
    )


def representation_to_root_branch(representation, features=None, operations=None, max_iter=500, verbose=0):
    if features is None:
        features = alphabet

    if operations is None:
        operations = DEFAULT_OPERATIONS

    # bad syntax checks
    par_stack = []
    no_open, no_closed = (0, 0)
    for c in representation:
        if c in ['(', ')']:
            par_stack.append(c)

        if len(par_stack) >= 2 and par_stack[-2] == '(' and par_stack[-1] == ')':
            par_stack.pop()
            par_stack.pop()

        if c == '(':
            no_open += 1
        elif c == ')':
            no_closed += 1

    if no_open != no_closed:
        raise dfjss_exceptions.BadSyntaxRepresentationError(f"Representation \'{representation}\' has different amounts of open and closed parentheses")

    if len(par_stack) > 0:
        raise dfjss_exceptions.BadSyntaxRepresentationError(f"Representation \'{representation}\' has ill formed parentheses")

    def begins_with(containing_string, contained_string):
        containing_string = str(containing_string)
        contained_string = str(contained_string)
        return (len(containing_string) >= len(contained_string)) and (
                containing_string[0:len(contained_string)] == contained_string)

    # crumbs: list of elements of the expression (features, operations, parentheses).
    # The elements will progressively be reduced by evaluating expressions inside parentheses,
    # and replacing them with their equivalent PriorityFunctionBranch.
    # The procedure ends when a single crumb is left,
    # which is the final PriorityFunctionBranch representing the whole expression.
    crumbs = []

    i = 0
    I = len(representation)

    # construct crumbs list
    while i < I:
        c = representation[i]
        found_anything = False

        # is c a parenthesis?
        if c in ['(', ')']:
            crumbs.append(c)
            found_anything = True
            i += 1

        # is c the first character of a feature?
        if not found_anything:
            for f in features:
                if begins_with(representation[i:], f):
                    crumbs.append(f)
                    found_anything = True
                    i += len(f)
                    break

        # is c the first character of an operation?
        if not found_anything:
            for op in operations:
                if begins_with(representation[i:], op):
                    crumbs.append(op)
                    found_anything = True
                    i += len(op)
                    break

        # is c the first character of a number?
        def is_number(s):
            try:
                float(s)
                return True
            except ValueError:
                return False

        if not found_anything:
            # if first character is a number, keep adding characters until it is no longer a number
            # note: since there will always be a closed parenthesis at the end of the number, any
            # well formed representation should have no problem
            if is_number(representation[i]):
                j = 0
                while (i + j) < I:
                    if is_number(representation[i:i + j + 2]):
                        j += 1
                    else:
                        crumbs.append(float(representation[i:i + j + 1]))
                        found_anything = True
                        i += j + 1
                        break

        if not found_anything:
            raise dfjss_exceptions.BadSyntaxRepresentationError(
                f"Unknown character \'{c}\' present which is not a parenthesis, the beginning of the name of a feature/operation, nor a number")

    def parentheses_locations():
        # creates a list of 2-tuples containing the index of open and closed parentheses
        # example: [(0, '('), (4, ')')]
        result = []

        for i, c in enumerate(crumbs):
            if c in ['(', ')']:
                result.append((i, c))

        return result

    # iterate through crumbs until only one crumb is left
    while len(crumbs) > 0 and max_iter > 0:
        if len(crumbs) == 1:
            if isinstance(crumbs[0], PriorityFunctionBranch):
                return crumbs[0]
            else:
                raise dfjss_exceptions.BadFinalCrumbRepresentationError(f"Final crumb {crumbs[0]} is not a PriorityFunctionBranch", crumb=crumbs[0])

        max_iter -= 1
        par_locs = parentheses_locations()

        for j, (i, c) in enumerate(par_locs):
            if j + 1 == len(par_locs):
                break

            # check things inside parentheses
            if par_locs[j][1] == '(' and par_locs[j + 1][1] == ')':
                start = par_locs[j][0]
                end = par_locs[j + 1][0]

                # syntax checks
                if (end - start) != 4:
                    raise dfjss_exceptions.BadSyntaxRepresentationError(f"Expression inside parentheses {''.join([str(crumb) for crumb in crumbs[start:end + 1]])} is too {'long' if (end - start) > 4 else 'short'}. Paretheses should just contain the first feature, one operation, and the second feature")

                sub_crumbs = crumbs[start + 1:end]

                new_branch = PriorityFunctionBranch(sub_crumbs[0], sub_crumbs[1], sub_crumbs[2])
                for _ in range(end - start + 1):
                    crumbs.pop(start)

                crumbs.insert(start, new_branch)

                # break the par_locs loop
                break

        if verbose > 1:
            print(crumbs)

    if max_iter == 0:
        raise dfjss_exceptions.TooManyIterationsRepresentationError(f"Maximum number of iterations reached (final crumbs state: {crumbs})", crumbs=crumbs)
    else:
        raise dfjss_exceptions.UnexpectedLoopEndRepresentationError(f"Loop ended unexpectedly with no result (final crumbs state: {crumbs})", crumbs=crumbs)

# DECISION RULE


class PriorityFunctionTreeDecisionRule(dfjss.BaseDecisionRule):
    priority_function_tree: PriorityFunctionTree

    def __init__(self, priority_function_tree):
        """

        :type priority_function_tree: PriorityFunctionTree
        """

        self.priority_function_tree = priority_function_tree

    def make_decision(self, warehouse):
        machines = warehouse.available_machines()
        operations = warehouse.operations_from_available_jobs()

        M, O = len(machines), len(operations)
        priority_values = np.fill(shape=(M, O), fill_value=np.nan)

        for m, machine in enumerate(machines):
            for o, operation in enumerate(operations):
                if not dfjss.machine_operation_compatible(machine, operation):
                    continue

                features = warehouse.warehouse_features | machine.features | operation.features

                priority_values[m, o] = self.priority_function_tree.run(features=features)

        if np.all(np.isnan(priority_values)):
            return dfjss.DecisionRuleOutput(success=False)

        m_max, o_max = np.unravel_index(np.argmax(a=priority_values, axis=None), (M, O))

        chosen_job = warehouse.job_of_operation(operations[o_max])

        return dfjss.DecisionRuleOutput(success=True,
                                        machine=machines[m_max],
                                        job=chosen_job,
                                        operation=operations[o_max])


# UNIT TESTS

# representation
ut_representation = "((a/(b+3.25))<c)"
ut_priority_function = representation_to_priority_function_tree(ut_representation)
ut_reconstruction = repr(ut_priority_function.root_branch)
assert ut_reconstruction == ut_representation, \
    f"Unit Test failed: reconstructing the reference priority function should have yielded {ut_representation}, but yielded {ut_reconstruction} instead"

# evaluation of expression
ut_features_values = {'a': 50, 'b': 6.75, 'c': 100}
ut_true_result = 5
ut_actual_result = ut_priority_function.run(ut_features_values)

assert ut_actual_result == ut_true_result, \
    f"Unit Test failed: evaluating the reference priority function with some reference values should have yielded {ut_true_result} as a result, but yielded {ut_actual_result} instead"
