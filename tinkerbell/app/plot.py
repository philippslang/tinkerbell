from matplotlib import pyplot as plt


def render_points(ax, points):
    xcoords = [point.x for point in points]
    ycoords = [point.y for point in points]
    ax.plot(xcoords, ycoords, 'x')


def plot_points(pointsiterable):
    fig = plt.figure() 
    ax = fig.add_subplot(111)
    for points in pointsiterable:
        render_points(ax, points)
    plt.show()