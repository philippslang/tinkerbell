import json


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


def point_coordinates(pts, idx=0):
    """
    Returns concatenated list of all x (idx=0) or y (idx=1) coordinates
    of the provided points.
    """
    return [pt[idx] for pt in pts]


def read_points(fname):
    with open(fname) as f:
        data = json.load(f)
    coords = data['coordinates']
    xcoords = coords['x']
    ycoords = coords['y']
    return [Point(x, y) for x, y in zip(xcoords, ycoords)]


def write_points(fname, pts):
    data = {'coordinates':{'x': point_coordinates(pts, 0), 'y': point_coordinates(pts, 1) }}
    with open(fname, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)
