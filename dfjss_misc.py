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
