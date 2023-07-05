def flatten(l):
    """

    :type l: list
    """
    return [item for sublist in l for item in sublist]


def dict_melt(d):
    """

    :type d: dict
    """
    return flatten(list(d.values()))
