"""
Integral
"""


def poly_integral(poly, C=0):
    """
    calculates the integral of a polynomial:
    """
    if (type(poly) is not list
            or len(poly) == 0 or not isinstance(C, (int, float))):
        return None
    res = [C]

    for index, coefficient in enumerate(poly):
        Q_val = coefficient / (1 + index)
        if Q_val.is_integer():
            res.append(int(Q_val))
        else:
            res.append(Q_val)

    return res
