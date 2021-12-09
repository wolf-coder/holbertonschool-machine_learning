"""
Integral
"""


def poly_integral(poly, C=0):
    """
    Calculate of a polynomial integral
    """
    if not isinstance(C, (float, int)):
        return None
    if type(poly) is not list or len(poly) == 0:
        return None
    if poly == [0]:
        return [C]

    Integral_Value = [C]

    for index in range(len(poly)):
        if not isinstance(poly[index], (int, float)):
            return None
        Coef = poly[index] / (index + 1)
        if Coef % 1 == 0:
            Integral_Value.append(int(Coef))
        else:
            Integral_Value.append(Coef)
    return Integral_Value
