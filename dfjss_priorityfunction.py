import numpy as np
from string import ascii_lowercase as alphabet
import numbers
from collections import deque

import dfjss_exceptions
import dfjss_objects as dfjss
import dfjss_misc as misc


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


DEFAULT_FEATURES = alphabet

def pf_add(x,y):
    return x + y

def pf_subtract(x, y):
    return x - y

def pf_multiply(x, y):
    return x * y

def pf_safediv(x, y):
    return x / y if y != 0 else x

def pf_power(x, y):
    return x ** y

def pf_min(x, y):
    return min(x, y)

def pf_max(x, y):
    return max(x, y)

DEFAULT_OPERATIONS = {
    "+": pf_add,
    "-": pf_subtract,
    "*": pf_multiply,
    "/": pf_safediv,
    "^": pf_power,
    "<": pf_min,
    ">": pf_max,
}

FORBIDDEN_CHARACTERS = ['(', ')']

def latex_feature_formatting(feature_name, remove_prefix=True, join_with_underscore=False):
    parts = feature_name.split("_")

    if remove_prefix:
        parts.pop(0)

    if join_with_underscore:
        return "\\texttt{{{feature}}}".format(feature="\\char`_".join(parts))

    return "\\texttt{{{feature}}}".format(feature=" ".join(parts))


DEFAULT_LATEX_FORMATTING = {
    "+": lambda x, y: "{x} + {y}".format(x=x, y=y),
    "-": lambda x, y: "{x} - {y}".format(x=x, y=y),
    "*": lambda x, y: "{x} \\cdot {y}".format(x=x, y=y),
    "/": lambda x, y: "\\frac{{{x}}}{{{y}}}".format(x=x, y=y),
    "^": lambda x, y: "{{{x}}} ^ {{{y}}}".format(x=x, y=y),
    "<": lambda x, y: "{x} \\land {y}".format(x=x, y=y),
    ">": lambda x, y: "{x} \\lor {y}".format(x=x, y=y),

    "feature": lambda x: latex_feature_formatting(x)
}

DEFAULT_LATEX_FEATURE_ABBREVIATIONS = {
    "number of ": "\\#"
}


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
        elif is_number(self.left_feature):
            left = misc.constant_format(self.left_feature)
        else:
            left = self.left_feature

        if isinstance(self.right_feature, PriorityFunctionBranch):
            right = repr(self.right_feature)
        elif is_number(self.right_feature):
            right = misc.constant_format(self.right_feature)
        else:
            right = self.right_feature

        return f"({left}{self.operation_character}{right})"

    def latex(self, formatting=None, abbreviations=None, parentheses=True, recursion_step=0):
        """
        Returns a representation in LaTeX syntax.

        :type formatting: dict[lambda Any, Any]
        :type abbreviations: dict[str]
        :type parentheses: bool
        :type recursion_step: int

        :param formatting: Specifies how operations get turned into syntax. If None, loads a default.
        :param abbreviations: Specifies how some redundant phrases get abbreviated (i.e. "number of " turns into "#")
        :param parentheses: Whether or not to have parentheses. Some "obviously not needed" parentheses will not be put.
        :param recursion_step: Internal parameter, do not pass as argument.
        :return: str
        """
        if formatting is None:
            formatting = DEFAULT_LATEX_FORMATTING

        if abbreviations is None:
            abbreviations = DEFAULT_LATEX_FEATURE_ABBREVIATIONS

        # \texttt{This is Inconsolata.}
        shell = formatting[self.operation_character]
        going_deeper = False

        if isinstance(self.left_feature, PriorityFunctionBranch):
            left = self.left_feature.latex(formatting=formatting, parentheses=parentheses, recursion_step=recursion_step+1)
            going_deeper = True
        elif is_number(self.left_feature):
            left = self.left_feature
        else:
            left = formatting["feature"](self.left_feature)

        if isinstance(self.right_feature, PriorityFunctionBranch):
            right = self.right_feature.latex(formatting=formatting, parentheses=parentheses, recursion_step=recursion_step+1)
            going_deeper = True
        elif is_number(self.right_feature):
            right = self.right_feature
        else:
            right = formatting["feature"](self.right_feature)

        if recursion_step == 0:
            for long, short in abbreviations.items():
                if type(left) is str:
                    left = left.replace(long, short)

                if type(right) is str:
                    right = right.replace(long, short)

        parentheses_condition = parentheses and going_deeper and recursion_step > 0

        return f"\\left({shell(left, right)}\\right)" if parentheses_condition else shell(left, right)

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

    def depth_of(self, inner_branch, current_depth=1):
        """
        Returns the depth at which the specified inner_branch is located.
        If the inner_branch is not found, returns -1.
        The root branch has depth of 1.

        (From ChatGPT)

        :param inner_branch: The target PriorityFunctionBranch to search for.
        :param current_depth: Internal parameter, do not pass as an argument.
        :return: int
        """
        if self is inner_branch:
            return current_depth

        left_depth = -1
        right_depth = -1

        if isinstance(self.left_feature, PriorityFunctionBranch):
            left_depth = self.left_feature.depth_of(inner_branch, current_depth + 1)

        if isinstance(self.right_feature, PriorityFunctionBranch):
            right_depth = self.right_feature.depth_of(inner_branch, current_depth + 1)

        # If the branch is found in either left or right subtree, return the maximum depth.
        if left_depth != -1 or right_depth != -1:
            return max(left_depth, right_depth)

        return -1  # If the branch is not found in the entire tree, return -1.

    def flatten(self):
        """
        Returns a list of all features and operations in the tree,
        depth-first, from left to right.

        :return: list[str]
        """
        result = []

        if isinstance(self.left_feature, PriorityFunctionBranch):
            result.extend(self.left_feature.flatten())
        else:
            result.append(self.left_feature)

        result.append(self.operation_character)

        if isinstance(self.right_feature, PriorityFunctionBranch):
            result.extend(self.right_feature.flatten())
        else:
            result.append(self.right_feature)

        return result

    def set_from_flattened(self, flattened_list, old_flattened=None):
        """
        Sets features and operations based on the given flattened list.
        If the flattened list does not match the branch's representation,
        raises a ValueError exception.

        :param flattened_list: The list representing a flattened PriorityFunctionBranch.
        :param old_flattened: Internal parameter, do not pass as an argument.
        :raises ValueError: If the flattened list does not match the branch's representation.
        """
        if old_flattened is None:
            old_flattened = []

        if isinstance(self.left_feature, PriorityFunctionBranch):
            self.left_feature.set_from_flattened(flattened_list=flattened_list, old_flattened=old_flattened)
        else:
            old_flattened.append(self.left_feature)
            self.left_feature = flattened_list[len(old_flattened) - 1]

        old_flattened.append(self.operation_character)
        self.operation_character = flattened_list[len(old_flattened) - 1]

        if isinstance(self.right_feature, PriorityFunctionBranch):
            self.right_feature.set_from_flattened(flattened_list=flattened_list, old_flattened=old_flattened)
        else:
            old_flattened.append(self.right_feature)
            self.right_feature = flattened_list[len(old_flattened) - 1]

        return old_flattened

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

        try:
            return operations[self.operation_character](left, right)
        except TypeError as error:
            raise TypeError(f"Could not do operation \'{self.operation_character}\' between {type(left)} ({self.left_feature} == {left}) and {type(right)} ({self.right_feature} == {right})")


class PriorityFunctionTree:
    """
    This object is used to construct mathematical expressions, given a set of
    features and operations that it can have. The expression has a tree-like
    structure, resembling the tree-like priority functions that are used in
    Job Shop Scheduling (JSS) problems.
    The expression can be evaluated when passing the values of the features.
    """
    root_branch: PriorityFunctionBranch

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

        """
        if self.features is not DEFAULT_FEATURES:
            result += f" (Custom Features: {self.features})"

        if self.operations is not DEFAULT_OPERATIONS:
            result += f" (Custom Operations: {self.operations})"
        """

        return result
    
    def latex(self, formatting=None, abbreviations=None, parentheses=True):
        return self.root_branch.latex(formatting=formatting, abbreviations=abbreviations, parentheses=parentheses)
    
    def count(self):
        return self.root_branch.count()

    def depth(self):
        return self.root_branch.depth()

    def depth_of(self, inner_branch):
        return self.root_branch.depth_of(inner_branch=inner_branch)

    def flatten(self):
        return self.root_branch.flatten()

    def set_from_flattened(self, flattened_list):
        return self.root_branch.set_from_flattened(flattened_list=flattened_list)

    def run(self, features):
        return self.root_branch.run(self.operations, features)

    def get_copy(self):
        return representation_to_priority_function_tree(
                representation=repr(self.root_branch),
                features=self.features,
                operations=self.operations)


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


def representation_to_crumbs(representation, features=None, operations=None):
    no_features_given = False
    if features is None:
        features = alphabet
        no_features_given = True

    no_operations_given = False
    if operations is None:
        operations = DEFAULT_OPERATIONS
        no_operations_given = True

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
            for f in sorted(features, key=lambda x: len(x), reverse=True):
                if begins_with(representation[i:], f):
                    crumbs.append(f)
                    found_anything = True
                    i += len(f)
                    break

        # is c the first character of a number?
        if not found_anything:
            if c == "{":
                j = 0
                while (i + j) < I:
                    if representation[i + j] != "}":
                        j += 1
                    else:
                        crumbs.append(float(representation[i+1:i + j]))
                        found_anything = True
                        i += j + 1
                        break
            """
            # if first character is a number, keep adding characters until it is no longer a number
            # note: since there will always be a closed parenthesis at the end of the number, any
            # well formed representation should have no problem
            if is_number(representation[i]) or (i > 0 and (representation[i-1] in operations or representation[i-1] == "(") and representation[i] == '-' and is_number(representation[i:i+2])):
                j = 0
                while (i + j) < I:
                    if is_number(representation[i:i + j + 2]):
                        j += 1
                    else:
                        crumbs.append(float(representation[i:i + j + 1]))
                        found_anything = True
                        i += j + 1
                        break
            """

        # is c the first character of an operation?
        if not found_anything:
            for op in operations:
                if begins_with(representation[i:], op):
                    crumbs.append(op)
                    found_anything = True
                    i += len(op)
                    break

        if not found_anything:
            raise dfjss_exceptions.BadSyntaxRepresentationError(
                f"""Unknown character \'{c}\' present which is not a parenthesis, the beginning of the name of a feature/operation, nor a number.
                **Could it be that custom features/operations were not given?**
                (Features: {'Not custom (defaulting to alphabet)' if no_features_given else 'Custom'}, Operations: {'Not custom (defaulting to +-*/^)' if no_operations_given else 'Custom'})
                Representation: {representation}""")

    return crumbs


def crumbs_parenthesis_locations(crumbs):
    # creates a list of 2-tuples containing the index of open and closed parentheses
    # example: [(0, '('), (4, ')')]
    result = []

    for i, c in enumerate(crumbs):
        if c in ['(', ')']:
            result.append((i, c))

    return result


def crumbs_to_root_branch(crumbs, max_iter=500, verbose=0):

    def parentheses_locations():
        return crumbs_parenthesis_locations(crumbs)

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


def representation_to_root_branch(representation, features=None, operations=None, max_iter=500, verbose=0):
    if features is None:
        features = alphabet

    if operations is None:
        operations = DEFAULT_OPERATIONS

    crumbs = representation_to_crumbs(representation=representation,
                                      features=features,
                                      operations=operations)

    return crumbs_to_root_branch(crumbs=crumbs, max_iter=max_iter, verbose=verbose)


def is_representation_valid(representation, features=None, operations=None, max_iter=500):
    try:
        representation_to_root_branch(representation, features, operations, max_iter)
    except (dfjss_exceptions.ForbiddenCharacterError,
            dfjss_exceptions.BadSyntaxRepresentationError,
            dfjss_exceptions.TooManyIterationsRepresentationError,
            dfjss_exceptions.UnexpectedLoopEndRepresentationError,
            dfjss_exceptions.BadFinalCrumbRepresentationError):
        return False
    except Exception as excp:
        raise excp

    return True

# DECISION RULE


class PriorityFunctionTreeDecisionRule(dfjss.BaseDecisionRule):
    priority_function_tree: PriorityFunctionTree

    def __init__(self, priority_function_tree):
        """

        :type priority_function_tree: PriorityFunctionTree
        """

        self.priority_function_tree = priority_function_tree

    def make_decisions(self, warehouse, allow_wait=True, values_offset=0., epsilon=5e-12):
        compatible_pairs = warehouse.compatible_pairs()

        if len(compatible_pairs) <= 0:
            return dfjss.DecisionRuleOutput(success=False)

        priority_values = np.array([self.priority_function_tree.run(
            features=warehouse.all_features_of_compatible_pair(machine=machine, job=job))
            for machine, job in compatible_pairs],
            dtype=np.float64) + values_offset

        remaining = np.full(shape=len(compatible_pairs), fill_value=True, dtype=bool)

        pairs = []

        while np.any(remaining) and (not allow_wait or np.any(np.bitwise_and(~np.isnan(priority_values), priority_values >= -epsilon, casting="same_kind"))):
            index_max = np.nanargmax(priority_values)

            pairs.append(compatible_pairs[index_max])

            remaining = np.bitwise_and(remaining,
                                       [(machine != compatible_pairs[index_max][0] and job != compatible_pairs[index_max][1])
                                        for i, (machine, job) in enumerate(compatible_pairs)], casting="same_kind")

            priority_values[~remaining] = np.nan

        return dfjss.DecisionRuleOutput(success=True, pairs=pairs)


# UNIT TESTS

# representation
ut_representation = "((a/(b+{3.25}))<c)"
ut_priority_function = representation_to_priority_function_tree(ut_representation)
ut_reconstruction = repr(ut_priority_function.root_branch)
assert ut_reconstruction == ut_representation, \
    f"Unit Test failed: reconstructing the reference priority function should have yielded {ut_representation}, but yielded {ut_reconstruction} instead"

#print(ut_priority_function.latex())

# evaluation of expression
ut_features_values = {'a': 50, 'b': 6.75, 'c': 100}
ut_true_result = 5
ut_actual_result = ut_priority_function.run(ut_features_values)

assert ut_actual_result == ut_true_result, \
    f"Unit Test failed: evaluating the reference priority function with some reference values should have yielded {ut_true_result} as a result, but yielded {ut_actual_result} instead"
