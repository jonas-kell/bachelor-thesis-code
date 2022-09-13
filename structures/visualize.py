import matplotlib.pyplot as plt
import math
import time
import numpy as np

from helpers.neighbors import (
    trigonal_hexagonal_nr_lattice_sites,
    trigonal_hexagonal_index_to_qr,
    hexagonal_nr_lattice_sites,
    hexagonal_index_to_qr,
)

from lattice_parameter_resolver import resolve_lattice_parameters, LatticeParameters

point_distance = 0.1
highlight_display_index = -1
svg_size_in_mm = 60


# Draw a point based on the x, y axis value.
def draw_point(x, y, label="", width_x=1, width_y=1, c="#0000aa"):
    plt.scatter(x, y, s=30, c=c)
    plt.text(
        x - 0.01 * point_distance * width_x,
        y - 0.03 * point_distance * width_y,
        str(label),
    )


def draw_svg_point(svg, x, y, label="", width_x=1, width_y=1, c="#0000aa"):
    svg.write(
        f"""    <g>
        <ellipse
            style="fill:{c};stroke-width:0.8"
            cx="{x}"
            cy="{y}"
            rx="{0.05* point_distance}"
            ry="{0.05* point_distance}" />
        <text
            transform="scale(1)"
            style="text-anchor: middle;alignment-baseline: central;font-size:{0.25 * point_distance};font-family:'Linux Libertine O';white-space:pre;fill:#000000;stroke-width:4">
            <tspan x="{x}" y="{y+0.23* point_distance}">{str(label)}</tspan>
        </text>
    </g>
"""
    )


# iterate over list of points to draw lattice
def draw_lattice(
    coords,
    lattice_parameters: LatticeParameters,
    width_x=1,
    width_y=1,
    output_svg=False,
):
    global highlight_display_index

    if output_svg:
        svg = open(f"./{time.time()}.svg", "w")

        bounds_test = np.array(coords)[:, 1:]
        bounds_min = bounds_test.min(axis=0)
        bounds_max = bounds_test.max(axis=0)

        svg.write(
            f"""<?xml version="1.0" encoding="UTF-8"?>

<svg
   width="{int(svg_size_in_mm * width_x / width_y)}mm"
   height="{svg_size_in_mm}mm"
   viewBox="{f"{bounds_min[0]-point_distance/2} {bounds_min[1]-point_distance/2} {bounds_max[0]-bounds_min[0]+point_distance} {bounds_max[1]-bounds_min[1]+point_distance}"}"
   version="1.1"
   id="svg5"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:svg="http://www.w3.org/2000/svg">
"""
        )

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

            if output_svg:
                draw_svg_point(
                    svg,
                    x,
                    bounds_max[1] + bounds_min[1] - y,
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
            else:
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

        if output_svg:
            break
        else:
            show()

    if output_svg:
        svg.write("</svg>\n")
        svg.close()


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


def draw_linear_lattice(
    size=1, periodic_bounds=False, random_swaps=0, output_svg=False
):
    coords = coords_linear_lattice(n=size)

    lattice_parameters = resolve_lattice_parameters(
        size=size, shape="linear", periodic=periodic_bounds, random_swaps=random_swaps
    )

    draw_lattice(
        coords=coords,
        lattice_parameters=lattice_parameters,
        width_x=size,
        width_y=size / 2,
        output_svg=output_svg,
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


def draw_cubic_lattice(size=1, periodic_bounds=False, random_swaps=0, output_svg=False):
    coords = coords_cubic_lattice(n=size + 1)

    lattice_parameters = resolve_lattice_parameters(
        size=size, shape="cubic", periodic=periodic_bounds, random_swaps=random_swaps
    )

    draw_lattice(
        coords=coords,
        lattice_parameters=lattice_parameters,
        width_x=size + 1,
        width_y=size + 1,
        output_svg=output_svg,
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


def draw_trigonal_square_lattice(
    size=1, periodic_bounds=False, random_swaps=0, output_svg=False
):
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
        output_svg=output_svg,
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


def draw_trigonal_diamond_lattice(
    size=1, periodic_bounds=False, random_swaps=0, output_svg=False
):
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
        output_svg=output_svg,
    )


def coords_trigonal_hexagonal_lattice(n=2):
    assert n > 1

    coords = []

    for index in range(trigonal_hexagonal_nr_lattice_sites(n)):
        q, r = trigonal_hexagonal_index_to_qr(index, n)

        x, y = cube_coordinates_to_cartesian_coordinates(q=q, r=r)

        coords.append((index, x, y))

    return coords


def draw_trigonal_hexagonal_lattice(
    size=1, periodic_bounds=False, random_swaps=0, output_svg=False
):
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
        output_svg=output_svg,
    )


def coords_hexagonal_lattice(n=1):
    assert n > 0

    coords = []

    for index in range(hexagonal_nr_lattice_sites(n)):
        q, r = hexagonal_index_to_qr(index, n)

        x, y = cube_coordinates_to_cartesian_coordinates(q=q, r=r)

        coords.append((index, x, y))

    return coords


def draw_hexagonal_lattice(
    size=1, periodic_bounds=False, random_swaps=0, output_svg=False
):
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
        output_svg=output_svg,
    )


def cube_coordinates_to_cartesian_coordinates(q, r):
    vertical_spacing = math.sqrt(3) / 2 * point_distance
    horizontal_spacing = point_distance

    # s = 0 - q - r

    x = q * horizontal_spacing + r * horizontal_spacing / 2
    y = -r * vertical_spacing

    return x, y


if __name__ == "__main__":
    # draw_linear_lattice(6, False, 0, False)
    draw_cubic_lattice(4, False, 0, False)
    # draw_trigonal_square_lattice(4, False, 0, False)
    # draw_trigonal_diamond_lattice(4, False, 0, False)
    # draw_trigonal_hexagonal_lattice(4, False, 0, False)
    # draw_hexagonal_lattice(3, True, 0, False)
