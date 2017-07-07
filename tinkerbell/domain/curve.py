import numpy as np
import scipy.interpolate as spint
import unittest


class Curve:
    def __init__(self, t, c, k):
        """
        The knots (all), coefficients and degree of the spline.
        """
        self._curve = spint.BSpline(t, c, int(k))

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

    def to_flat(self):
        """
        Returns a numpy array representing the state of the curve,
        i.e. serialization function.
        """
        return np.r_[[self._curve.k], [len(self._curve.t)], self._curve.t, self._curve.c]

    @staticmethod
    def from_flat(flatdata):
        """
        Returns a curve from its flat representation.
        """
        num_knots = int(flatdata[1])
        return Curve(flatdata[2:2+num_knots], flatdata[2+num_knots:], flatdata[0])

    def flat_header(self):
        """
        Returns tuple of strings denoting the columns of the flat representation.
        """
        return flat_header(len(self.t), len(self.c))


def flat_header_coefficients(num_cofficients):
    return ['c{:0>3d}'.format(i) for i in range(num_cofficients)]


def flat_header(num_knots, num_cofficients):
    """
    Returns tuple of strings denoting the columns of the flat representation.
    """
    knot_names = ['t{:0>3d}'.format(i) for i in range(num_knots)]
    coefficient_names = flat_header_coefficients(num_cofficients)
    return ('degree', 'num_knots', *knot_names, *coefficient_names)


class TestCurve(unittest.TestCase):

    def setUp(self):
        pass

    def test_serialization(self):  
        curve = Curve([0., 0., 0., 9.0, 10.0, 11.0, 36.6, 50.0, 50.0, 50.0], \
            [51.6, 30.5, 20.4, 25.5, 10.9, 11.2, 10.01135478, 0., 0., 0.], 2) 

        curve_flat = curve.to_flat()
        curve_clone = Curve.from_flat(curve_flat)

        self.assertEqual(isinstance(curve_clone, Curve), True)
        self.assertEqual(curve_clone.k, curve.k)
        self.assertEqual(np.array_equal(curve_clone.t, curve.t), True)
        self.assertEqual(np.array_equal(curve_clone.c, curve.c), True)
        print(curve.flat_header())
        self.assertEqual(len(curve_flat), len(curve.flat_header()))
            


if __name__ == '__main__':
    unittest.main()