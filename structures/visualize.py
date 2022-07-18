import matplotlib.pyplot as plt
import math

from neighbors import (
    square_lattice_get_nn_indices,
    square_lattice_get_nnn_indices,
    triangular_square_lattice_get_nn_indices,
    triangular_square_lattice_get_nnn_indices,
    triangular_diamond_lattice_get_nn_indices,
    triangular_diamond_lattice_get_nnn_indices,
)

point_distance = 0.1
highlight_index = -1

# Draw a point based on the x, y axis value.
def draw_point(x, y, label="", most_point_distances=1, c="#0000aa"):
    plt.scatter(x, y, s=30, c=c)
    plt.text(
        x - 0.01 * point_distance * most_point_distances,
        y - 0.03 * point_distance * most_point_distances,
        str(label),
    )


# iterate over list of points to draw lattice
def draw_lattice(coords, n, nn_function=None, nnn_function=None, periodic_bounds=False):
    global highlight_index

    while True:
        init(coords)

        if nn_function is None:
            nn_indices = []
        else:
            nn_indices = nn_function(highlight_index, n, periodic_bounds)

        if nnn_function is None:
            nnn_indices = []
        else:
            nnn_indices = nnn_function(highlight_index, n, periodic_bounds)

        for index, x, y in coords:
            draw_point(
                x,
                y,
                index,
                most_point_distances=n,
                c="#aa0000"
                if index == highlight_index
                else (
                    "#00aa00"
                    if index in nn_indices
                    else ("#aaaa00" if index in nnn_indices else "#0000aa")
                ),
            )

        show()


# init plot surface
def init(coords):
    fig = plt.figure(figsize=(9, 9))

    def onclick(event):
        global highlight_index

        ix, iy = event.xdata, event.ydata

        print("x = %f, y = %f" % (ix, iy))

        closest_index = -1
        closest_x = -100000
        closest_y = -100000

        for index, x, y in coords:
            if ((x - ix) ** 2 + (y - iy) ** 2) < (
                (closest_x - ix) ** 2 + (closest_y - iy) ** 2
            ):
                closest_index = index
                closest_x = x
                closest_y = y

        print("closest index: %d" % (closest_index))
        highlight_index = closest_index

        plt.close()  # "force redraw"

    fig.canvas.mpl_connect("button_press_event", onclick)


# trigger displaying of plot surface
def show():
    plt.margins(x=2 * point_distance, y=2 * point_distance)
    plt.show()


def coords_square_lattice(n=1):
    assert n > 0
    coords = []
    index = 0
    for j in range(n):
        for i in range(n):
            coords.append(
                (
                    index,
                    i * point_distance,
                    -1 * j * point_distance,
                )
            )
            index += 1

    return coords


def draw_square_lattice(n=1, periodic_bounds=False):
    coords = coords_square_lattice(n=n)

    draw_lattice(
        coords,
        n,
        nn_function=square_lattice_get_nn_indices,
        nnn_function=square_lattice_get_nnn_indices,
        periodic_bounds=periodic_bounds,
    )


def coords_triangular_square_lattice(n=1):
    assert n > 0
    coords = []

    y_step = math.sin(60 / 180 * math.pi) * point_distance
    x_offset = point_distance / 2.0

    y_value = 0

    size = 2 * n

    index = 0
    for i in range(size):  # row
        for j in range(size):  # "col"

            if i % 2 == 0:
                # "right" row
                coords.append((index, j * point_distance, y_value))
            else:
                # "left" row
                coords.append((index, j * point_distance - x_offset, y_value))

            index += 1
        y_value -= y_step

    return coords


def draw_triangular_square_lattice(n=1, periodic_bounds=False):
    coords = coords_triangular_square_lattice(n=n)

    draw_lattice(
        coords,
        2 * n,
        nn_function=triangular_square_lattice_get_nn_indices,
        nnn_function=triangular_square_lattice_get_nnn_indices,
        periodic_bounds=periodic_bounds,
    )


def coords_triangular_diamond_lattice(n=1):
    assert n > 0
    coords = []

    y_step = math.sin(60 / 180 * math.pi) * point_distance

    y_value = -n * y_step
    x_offset = point_distance / 2.0

    index = 0
    for i in range(-n, n + 1):  # row
        for j in range(n - abs(i) + 1):
            coords.append((index, abs(i) * x_offset + j * point_distance, y_value))
            index += 1

        y_value -= y_step

    return coords


def draw_triangular_diamond_lattice(n=1, periodic_bounds=False):
    coords = coords_triangular_diamond_lattice(n=n)

    draw_lattice(
        coords,
        2 * n,
        nn_function=triangular_diamond_lattice_get_nn_indices,
        nnn_function=triangular_diamond_lattice_get_nnn_indices,
        periodic_bounds=periodic_bounds,
    )


if __name__ == "__main__":
    # draw_square_lattice(6, True)
    # draw_triangular_square_lattice(3, True)

    draw_triangular_diamond_lattice(6, True)
