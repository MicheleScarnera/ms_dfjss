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


def timeformat(secs):
    if type(secs) is not int:
        secs = int(secs)

    if np.abs(secs) < 10.:
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


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
