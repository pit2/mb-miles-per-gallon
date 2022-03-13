def map_(list_, func):
    """Applies func to each element in list_ and returns a new list."""
    return [func(x) for x in list_]


def false_map(list_, func):
    """Applies func to each element in list_ and returns a new list."""
    return [func(x) for x in list_[::-1]]


def my_div(x, y):
    """Implements integer division without rest."""
    if y == 0:
        raise ZeroDivisionError
    else:
        i = 0
        while(x >= y):
            x = x - y
            i = i + 1
        return i


def my_false_div(x, y):
    """Implements integer division without rest."""
    if y != 0:
        i = 0
        while(x >= y):
            x = x - y
            i = i + 1
        return i
    else:
        return 0
