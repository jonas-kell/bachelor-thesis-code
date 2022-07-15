import matplotlib.pyplot as plt
import math

point_distance = 0.1
highlight_index = -1

# Draw a point based on the x, y axis value.
def draw_point(x, y, label="", most_point_distances=1, highlight=False):
    plt.scatter(x, y, s=30, c=("#0000aa" if not highlight else "#00aa00"))
    plt.text(
        x - 0.01 * point_distance * most_point_distances,
        y - 0.03 * point_distance * most_point_distances,
        str(label),
    )


# iterate over list of points to draw lattice
def draw_lattice(coords, most_point_distances):
    global highlight_index

    while True:
        init(coords)

        for index, x, y in coords:
            draw_point(
                x,
                y,
                index,
                most_point_distances=most_point_distances,
                highlight=index == highlight_index,
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


def draw_square_lattice(n=1):
    coords = coords_square_lattice(n=n)

    draw_lattice(coords, n)


def coords_triangular_lattice(n=1):
    assert n > 0
    coords = []

    y_step = math.sin(60 / 180 * math.pi) * point_distance

    y_value = 0

    index = 0
    for i in range(n):  # row
        if i % 2 == 0:
            # odd row
            offsets = list(range(-(i // 2), (i // 2) + 1))
        else:
            # even row
            offsets = [i + 0.5 for i in range(-((i + 1) // 2), (i + 1) // 2)]

        for j in offsets:
            coords.append((index, j * point_distance, y_value))
            index += 1

        y_value -= y_step

    return coords


def draw_triangular_lattice(n=1):
    coords = coords_triangular_lattice(n=n)

    draw_lattice(coords, n)


if __name__ == "__main__":
    # draw_square_lattice(3)
    draw_triangular_lattice(5)
