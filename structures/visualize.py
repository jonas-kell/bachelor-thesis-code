import matplotlib.pyplot as plt
import math

point_distance = 0.1

# Draw a point based on the x, y axis value.
def draw_point(x, y, label="", most_point_distances=1):
    plt.scatter(x, y, s=30, c="#0000aa")
    plt.text(
        x - 0.01 * point_distance * most_point_distances,
        y - 0.03 * point_distance * most_point_distances,
        str(label),
    )


def draw_square_lattice(n=1):
    init()

    index = 0
    for j in range(n):
        for i in range(n):
            draw_point(
                i * point_distance,
                -1 * j * point_distance,
                index,
                most_point_distances=n,
            )
            index += 1

    show()


def draw_triangular_lattice(n=1):
    init()

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
            draw_point(j * point_distance, y_value, index, most_point_distances=n)
            index += 1

        y_value -= y_step

    show()


def show():
    plt.margins(x=2 * point_distance, y=2 * point_distance)
    plt.show()


def init():
    plt.figure(figsize=(9, 9))


if __name__ == "__main__":
    # draw_square_lattice(8)
    # draw_triangular_lattice(5)
    draw_hexagonal_lattice(5)
