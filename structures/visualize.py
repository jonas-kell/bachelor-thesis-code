import matplotlib.pyplot as plt
import math

from helpers.neighbors import (
    trigonal_hexagonal_nr_lattice_sites,
    trigonal_hexagonal_index_to_qr,
    hexagonal_nr_lattice_sites,
    hexagonal_index_to_qr,
)

from lattice_parameter_resolver import resolve_lattice_parameters, LatticeParameters

point_distance = 0.1
highlight_display_index = -1

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
    lattice_parameters: LatticeParameters,
    width_x=1,
    width_y=1,
):
    global highlight_display_index

    while True:
        init(coords, width_x, width_y)

        for index in range(lattice_parameters["nr_sites"]):

            display_index = lattice_parameters["display_indices_lookup"][index]
            highlight_index = (
                lattice_parameters["display_indices"][highlight_display_index]
                if highlight_display_index >= 0
                else -1
            )
            x = coords[display_index][1]
            y = coords[display_index][2]

            nn_indices = (
                lattice_parameters["nn_interaction_indices"][highlight_index]
                if highlight_index >= 0
                else []
            )
            nnn_indices = (
                lattice_parameters["nnn_interaction_indices"][highlight_index]
                if highlight_index >= 0
                else []
            )

            draw_point(
                x,
                y,
                index,  # label
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
        global highlight_display_index

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

        print("closest display index to highlight: %d" % (closest_index))
        highlight_display_index = closest_index

        plt.close()  # "force redraw"

    fig.canvas.mpl_connect("button_press_event", onclick)


# trigger displaying of plot surface
def show():
    plt.margins(x=2 * point_distance, y=2 * point_distance)
    plt.show()


def coords_linear_lattice(n=1):
    assert n > 0

    coords = []
    for index in range(n):
        coords.append(
            (
                index,
                index * point_distance,
                0,
            )
        )

    return coords


def draw_linear_lattice(size=1, periodic_bounds=False, random_swaps=0):
    coords = coords_linear_lattice(n=size)

    lattice_parameters = resolve_lattice_parameters(
        size=size, shape="linear", periodic=periodic_bounds, random_swaps=random_swaps
    )

    draw_lattice(
        coords=coords,
        lattice_parameters=lattice_parameters,
        width_x=size,
        width_y=size / 2,
    )


def coords_cubic_lattice(n=1):
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


def draw_cubic_lattice(size=1, periodic_bounds=False, random_swaps=0):
    coords = coords_cubic_lattice(n=size + 1)

    lattice_parameters = resolve_lattice_parameters(
        size=size, shape="cubic", periodic=periodic_bounds, random_swaps=random_swaps
    )

    draw_lattice(
        coords=coords,
        lattice_parameters=lattice_parameters,
        width_x=size + 1,
        width_y=size + 1,
    )


def coords_trigonal_square_lattice(n=2):
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


def draw_trigonal_square_lattice(size=1, periodic_bounds=False, random_swaps=0):
    coords = coords_trigonal_square_lattice(n=2 * size)

    lattice_parameters = resolve_lattice_parameters(
        size=size,
        shape="trigonal_square",
        periodic=periodic_bounds,
        random_swaps=random_swaps,
    )

    draw_lattice(
        coords=coords,
        lattice_parameters=lattice_parameters,
        width_x=2 * size - 0.5,
        width_y=(2 * size - 1) * math.sin(60.0 / 180.0 * math.pi),
    )


def coords_trigonal_diamond_lattice(n=1):
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


def draw_trigonal_diamond_lattice(size=1, periodic_bounds=False, random_swaps=0):
    coords = coords_trigonal_diamond_lattice(n=size)

    lattice_parameters = resolve_lattice_parameters(
        size=size,
        shape="trigonal_diamond",
        periodic=periodic_bounds,
        random_swaps=random_swaps,
    )

    draw_lattice(
        coords=coords,
        lattice_parameters=lattice_parameters,
        width_x=size,
        width_y=2 * size * math.sin(60.0 / 180.0 * math.pi),
    )


def coords_trigonal_hexagonal_lattice(n=2):
    assert n > 1

    coords = []

    for index in range(trigonal_hexagonal_nr_lattice_sites(n)):
        q, r = trigonal_hexagonal_index_to_qr(index, n)

        x, y = cube_coordinates_to_cartesian_coordinates(q=q, r=r)

        coords.append((index, x, y))

    return coords


def draw_trigonal_hexagonal_lattice(size=1, periodic_bounds=False, random_swaps=0):
    coords = coords_trigonal_hexagonal_lattice(n=size + 1)

    lattice_parameters = resolve_lattice_parameters(
        size=size,
        shape="trigonal_hexagonal",
        periodic=periodic_bounds,
        random_swaps=random_swaps,
    )

    draw_lattice(
        coords=coords,
        lattice_parameters=lattice_parameters,
        width_x=2 * size,
        width_y=(2 * size) * math.sin(60.0 / 180.0 * math.pi),
    )


def coords_hexagonal_lattice(n=1):
    assert n > 0

    coords = []

    for index in range(hexagonal_nr_lattice_sites(n)):
        q, r = hexagonal_index_to_qr(index, n)

        x, y = cube_coordinates_to_cartesian_coordinates(q=q, r=r)

        coords.append((index, x, y))

    return coords


def draw_hexagonal_lattice(size=1, periodic_bounds=False, random_swaps=0):
    coords = coords_hexagonal_lattice(n=size)

    lattice_parameters = resolve_lattice_parameters(
        size=size,
        shape="hexagonal",
        periodic=periodic_bounds,
        random_swaps=random_swaps,
    )

    draw_lattice(
        coords=coords,
        lattice_parameters=lattice_parameters,
        width_x=(2 * size) - 1 + 2 * size * math.cos(60.0 / 180.0 * math.pi),
        width_y=(2 * size) - 1 + 2 * size * math.cos(60.0 / 180.0 * math.pi),
    )


def cube_coordinates_to_cartesian_coordinates(q, r):
    vertical_spacing = math.sqrt(3) / 2 * point_distance
    horizontal_spacing = point_distance

    # s = 0 - q - r

    x = q * horizontal_spacing + r * horizontal_spacing / 2
    y = -r * vertical_spacing

    return x, y


if __name__ == "__main__":
    draw_linear_lattice(20, False, -1)
    # draw_cubic_lattice(4, False, -1)
    # draw_trigonal_square_lattice(4, False, -1)
    # draw_trigonal_diamond_lattice(4, False, -1)
    # draw_trigonal_hexagonal_lattice(4, False, -1)
    # draw_hexagonal_lattice(3, True, -1)
