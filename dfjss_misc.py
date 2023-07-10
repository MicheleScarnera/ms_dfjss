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
    if type(secs) is not int: secs = int(secs)
    # Formats an integer secs into a HH:MM:SS format.
    return f"{str(secs // 3600).zfill(2)}:{str((secs // 60) % 60).zfill(2)}:{str(secs % 60).zfill(2)}"
