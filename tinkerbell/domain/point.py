import json
import numpy as np

class Point:
    def __init__(self, x, y):
        self.coordinates = [x, y]

    def __getitem__(self, key):
        return self.coordinates[key]

    @property
    def x(self):
        return self.coordinates[0]

    @property
    def y(self):
        return self.coordinates[1]


def point_coordinates(pts, idx=None):
    """
    Returns concatenated list of all x (idx=0) or y (idx=1), or (x, y) (idx=None) coordinates
    of the provided points.
    """
    if idx is not None:
        return np.array([pt[idx] for pt in pts])
    else:
        return np.array([pt[0] for pt in pts]), np.array([pt[1] for pt in pts])

def from_coordinates(xcoords, ycoords):
    """
    Returns list of points from coordinate list.
    """
    return [Point(x, y) for x, y in zip(xcoords, ycoords)]


def read_points(fname):
    with open(fname) as f:
        data = json.load(f)
    coords = data['coordinates']
    xcoords = coords['x']
    ycoords = coords['y']
    return [Point(x, y) for x, y in zip(xcoords, ycoords)]


def write_points(fname, pts):
    data = {'coordinates':{'x': list(point_coordinates(pts, 0)), 'y': list(point_coordinates(pts, 1)) }}
    with open(fname, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)
