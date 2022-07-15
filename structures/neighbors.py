def square_lattice_get_nn_indices(index, n, periodic_bounds=False):
    neighbors = []

    if not 0 <= index < n * n:
        return neighbors

    x = index % n
    y = index // n

    if periodic_bounds:
        neighbors.append(((y + 1) % n) * n + ((x) % n))
        neighbors.append(((y - 1) % n) * n + ((x) % n))
        neighbors.append(((y) % n) * n + ((x + 1) % n))
        neighbors.append(((y) % n) * n + ((x - 1) % n))
    else:
        for xi, yi in [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]:
            if 0 <= xi < n and 0 <= yi < n:
                neighbors.append(yi * n + xi)

    return neighbors


def square_lattice_get_nnn_indices(index, n, periodic_bounds=False):
    neighbors = []

    if not 0 <= index < n * n:
        return neighbors

    x = index % n
    y = index // n

    if periodic_bounds:
        neighbors.append(((y + 1) % n) * n + ((x + 1) % n))
        neighbors.append(((y - 1) % n) * n + ((x + 1) % n))
        neighbors.append(((y + 1) % n) * n + ((x - 1) % n))
        neighbors.append(((y - 1) % n) * n + ((x - 1) % n))
    else:
        for xi, yi in [
            (x + 1, y + 1),
            (x - 1, y + 1),
            (x + 1, y - 1),
            (x - 1, y - 1),
        ]:
            if 0 <= xi < n and 0 <= yi < n:
                neighbors.append(yi * n + xi)

    return neighbors


if __name__ == "__main__":
    print(square_lattice_get_nn_indices(4, 3))
    print(square_lattice_get_nn_indices(0, 3))
