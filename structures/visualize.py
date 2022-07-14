from turtle import color
import matplotlib.pyplot as plt

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
            draw_point(i * point_distance, -1 * j * point_distance, index, n)
            index += 1

    show()


def show():
    plt.margins(x=2 * point_distance, y=2 * point_distance)
    plt.show()


def init():
    plt.figure(figsize=(9, 9))


if __name__ == "__main__":
    draw_square_lattice(8)
