def square_lattice_get_nn_indices(index, n, periodic_bounds=False):
    neighbors = []

    if not 0 <= index < n * n:
        return neighbors

    x = index % n
    y = index // n

    for xi, yi in [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x, y - 1),
    ]:
        if (0 <= xi < n and 0 <= yi < n) or periodic_bounds:
            neighbors.append((yi % n) * n + (xi % n))

    return neighbors


def square_lattice_get_nnn_indices(index, n, periodic_bounds=False):
    neighbors = []

    if not 0 <= index < n * n:
        return neighbors

    x = index % n
    y = index // n

    for xi, yi in [
        (x + 1, y + 1),
        (x - 1, y + 1),
        (x + 1, y - 1),
        (x - 1, y - 1),
    ]:
        if (0 <= xi < n and 0 <= yi < n) or periodic_bounds:
            neighbors.append((yi % n) * n + xi % n)

    return neighbors


def triangular_square_lattice_get_nn_indices(index, n, periodic_bounds=False):
    assert n % 2 == 0

    neighbors = []

    if not 0 <= index < n * n:
        return neighbors

    x = index % n
    y = index // n

    for xi, yi in [
        (x + 1, y),
        (x - 1, y),
        (x, y + 1),
        (x + (1 if y % 2 == 0 else -1), y + 1),
        (x, y - 1),
        (x + (1 if y % 2 == 0 else -1), y - 1),
    ]:
        if (0 <= xi < n and 0 <= yi < n) or periodic_bounds:
            neighbors.append((yi % n) * n + xi % n)

    return neighbors


def triangular_square_lattice_get_nnn_indices(index, n, periodic_bounds=False):
    assert n % 2 == 0

    neighbors = []

    if not 0 <= index < n * n:
        return neighbors

    x = index % n
    y = index // n

    for xi, yi in [
        (x, y + 2),
        (x, y - 2),
        (x + (-1 if y % 2 == 0 else -2), y + 1),
        (x + (2 if y % 2 == 0 else 1), y + 1),
        (x + (-1 if y % 2 == 0 else -2), y - 1),
        (x + (2 if y % 2 == 0 else 1), y - 1),
    ]:
        if (0 <= xi < n and 0 <= yi < n) or periodic_bounds:
            neighbors.append((yi % n) * n + xi % n)

    return neighbors


if __name__ == "__main__":
    print(square_lattice_get_nn_indices(4, 3))
    print(square_lattice_get_nn_indices(0, 3))
