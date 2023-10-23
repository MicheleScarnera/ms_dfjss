import numpy as np

def flatten(l):
    """

    :type l: list
    """
    return [item for sublist in l for item in sublist]


def dict_flatten_values(d):
    """

    :type d: dict
    """
    return flatten(list(d.values()))


def dictformat(dict) -> str:
    """

    :param dict:
    :return: Returns a better-looking formatting for a dictionary.
    """

    result = []
    for key, value in dict.items():
        result.append(f"{key} -> {value}")

    result = ",\n".join(result)
    return result


def constant_format(x):
    return f"{{{x}}}"


def large_number_format(x):
    log10 = np.log10(np.abs(x))

    if log10 >= 9:
        return f"{x / 1000000000:.1f}b"
    elif log10 >= 6:
        return f"{x / 1000000:.1f}m"
    elif log10 >= 3:
        return f"{x / 1000:.1f}k"
    else:
        return f"{x:.1f}"


def timeformat(secs):
    if np.abs(secs) < 60.:
        return f"{secs:.2f}s"
    else:
        return timeformat_hhmmss(secs)


def timeformat_hhmmss(secs):
    """
    Formats an integer secs into a HH:MM:SS format.

    :param secs:
    :return:
    """
    if type(secs) is not int:
        secs = int(secs)

    sign = np.sign(secs)
    secs = np.abs(secs)
    return f"{'' if sign >= 0 else '-'}{str(secs // 3600).zfill(2)}:{str((secs // 60) % 60).zfill(2)}:{str(secs % 60).zfill(2)}"


def timeleft(start, now, i, I):
    return (I - i) / i * (now - start)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def harmonic_number(n):
    """
    Returns an exact value of n-th harmonic number.
    http://en.wikipedia.org/wiki/Harmonic_number

    :type n: int
    :return np.double
    """

    if n <= 0:
        return 0.

    return np.sum([1. / i for i in range(1, n+1)], dtype=np.double)


def H(n):
    """
    Returns an exact value of n-th harmonic number.
    http://en.wikipedia.org/wiki/Harmonic_number

    :type n: float
    :return float
    """
    # Euler-Mascheroni constant
    gamma = 0.57721566490153286060651209008240243104215933593992
    return gamma + np.log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)


def begins_with(containing_string, contained_string):
    containing_string = str(containing_string)
    contained_string = str(contained_string)
    return (len(containing_string) >= len(contained_string)) and (
            containing_string[0:len(contained_string)] == contained_string)
