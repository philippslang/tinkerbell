import scipy.interpolate as spint


class Curve:
    def __init__(self, t, c, k):
        """
        The knots (all), coefficients and degree of the spline.
        """
        self._curve = spint.BSpline(t, c, k)

    def __call__(self, x):
        return self._curve(x)
    
    @property
    def t(self):
        return self._curve.t

    @property
    def k(self):
        return self._curve.k

    @property
    def c(self):
        return self._curve.c