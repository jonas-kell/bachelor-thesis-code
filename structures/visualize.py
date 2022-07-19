import matplotlib.pyplot as plt
import math

from neighbors import (
    square_lattice_get_nn_indices,
    square_lattice_get_nnn_indices,
    triangular_square_lattice_get_nn_indices,
    triangular_square_lattice_get_nnn_indices,
    triangular_diamond_lattice_get_nn_indices,
    triangular_diamond_lattice_get_nnn_indices,
    triangular_hexagonal_nr_lattice_sites,
    triangular_hexagonal_index_to_qr,
    triangular_hexagonal_lattice_get_nn_indices,
    triangular_hexagonal_lattice_get_nnn_indices,
)

point_distance = 0.1
highlight_index = -1

# Draw a point based on the x, y axis value.
def draw_point(x, y, label="", width_x=1, width_y=1, c="#0000aa"):
    plt.scatter(x, y, s=30, c=c)
    plt.text(
        x - 0.01 * point_distance * width_x,
        y - 0.03 * point_distance * width_y,
        str(label),
    )


# iterate over list of points to draw lattice
def draw_lattice(
    coords,
    n,
    width_x=1,
    width_y=1,
    nn_function=None,
    nnn_function=None,
    periodic_bounds=False,
):
    global highlight_index

    while True:
        init(coords, width_x, width_y)

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
                width_x=width_x,
                width_y=width_y,
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
def init(coords, width_x, width_y):
    fig = plt.figure(figsize=(9 * width_x / width_y, 9))

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


def draw_square_lattice(size=1, periodic_bounds=False):
    coords = coords_square_lattice(n=size)

    draw_lattice(
        coords,
        n=size,
        width_x=size,
        width_y=size,
        nn_function=square_lattice_get_nn_indices,
        nnn_function=square_lattice_get_nnn_indices,
        periodic_bounds=periodic_bounds,
    )


def coords_triangular_square_lattice(n=2):
    assert n > 0
    assert n % 2 == 0
    coords = []

    y_step = math.sin(60 / 180 * math.pi) * point_distance
    x_offset = point_distance / 2.0

    y_value = 0

    index = 0
    for i in range(n):  # row
        for j in range(n):  # "col"

            if i % 2 == 0:
                # "right" row
                coords.append((index, j * point_distance, y_value))
            else:
                # "left" row
                coords.append((index, j * point_distance - x_offset, y_value))

            index += 1
        y_value -= y_step

    return coords


def draw_triangular_square_lattice(size=1, periodic_bounds=False):
    coords = coords_triangular_square_lattice(n=2 * size)

    draw_lattice(
        coords,
        n=2 * size,
        width_x=2 * size - 0.5,
        width_y=(2 * size - 1) * math.sin(60.0 / 180.0 * math.pi),
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


def draw_triangular_diamond_lattice(size=1, periodic_bounds=False):
    coords = coords_triangular_diamond_lattice(n=size)

    draw_lattice(
        coords,
        n=size,
        width_x=size,
        width_y=2 * size * math.sin(60.0 / 180.0 * math.pi),
        nn_function=triangular_diamond_lattice_get_nn_indices,
        nnn_function=triangular_diamond_lattice_get_nnn_indices,
        periodic_bounds=periodic_bounds,
    )


def coords_triangular_hexagonal_lattice(n=2):
    assert n > 1

    coords = []

    for index in range(triangular_hexagonal_nr_lattice_sites(n)):
        q, r = triangular_hexagonal_index_to_qr(index, n)

        x, y = cube_coordinatex_to_cartesian_coordinates(q=q, r=r)

        coords.append((index, x, y))

    return coords


def draw_triangular_hexagonal_lattice(size=1, periodic_bounds=False):
    n = size + 1
    coords = coords_triangular_hexagonal_lattice(n=n)

    draw_lattice(
        coords,
        n=n,
        width_x=2 * size,
        width_y=(2 * size) * math.sin(60.0 / 180.0 * math.pi),
        nn_function=triangular_hexagonal_lattice_get_nn_indices,
        nnn_function=triangular_hexagonal_lattice_get_nnn_indices,
        periodic_bounds=periodic_bounds,
    )


def cube_coordinatex_to_cartesian_coordinates(q, r):
    vertical_spacing = math.sqrt(3) / 2 * point_distance
    horizontal_spacing = point_distance

    # s = 0 - q - r

    x = q * horizontal_spacing + r * horizontal_spacing / 2
    y = -r * vertical_spacing

    return x, y


if __name__ == "__main__":
    # draw_square_lattice(6, True)
    # draw_triangular_square_lattice(3, True)
    # draw_triangular_diamond_lattice(4, True)
    draw_triangular_hexagonal_lattice(4, True)
