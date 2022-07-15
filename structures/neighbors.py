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


def triangular_hexagonal_lattice_get_nn_indices(index, rows, periodic_bounds=False):
    assert rows % 2 == 0
    n = rows // 2

    neighbors = []

    if not 0 <= index < (n + 1) * (n + 1):
        return neighbors

    y, x = triag_hex_index_to_row_col(index, n)

    for xi, yi in [
        (x + (0 if y <= n else 1), y - 1),
        (x + (0 if y <= n else 1) - 1, y - 1),
        (x - 1, y),
        (x + 1, y),
        (x + (0 if y < n else -1), y + 1),
        (x + (0 if y < n else -1) + 1, y + 1),
    ]:
        neighbors.append(
            triag_hex_row_col_to_index(yi, xi, n, periodic_bounds=periodic_bounds)
        )

    # filter neg values (may be generated by `triag_hex_row_col_to_index` if periodic_bounds=False)
    neighbors = [i for i in neighbors if i >= 0]

    return neighbors


def triangular_hexagonal_lattice_get_nnn_indices(index, rows, periodic_bounds=False):

    return []


def triag_hex_index_to_row_col(index, n):
    row = 0
    col = 0

    for i in [n - abs(j) + 1 for j in range(-n, n + 1)]:
        if index >= i:
            index -= i
            row += 1
        else:
            col = index
            break

    return row, col


def triag_hex_row_col_to_index(row, col, n, periodic_bounds=True):
    index = 0

    if (
        row < 0
        or row > 2 * n
        or (row <= n and col > row)
        or (row > n and col > 2 * n - row)
    ):
        if periodic_bounds:
            # transform row/col into valid space
            pass
        else:
            # return invalid
            return -1

    for i in [n - abs(j) + 1 for j in range(-n, row - n)]:
        index += i

    index += col

    return index


if __name__ == "__main__":
    # print(square_lattice_get_nn_indices(4, 3))
    # print(square_lattice_get_nn_indices(0, 3))

    print(triag_hex_index_to_row_col(0, 2))
    print(triag_hex_index_to_row_col(1, 2))
    print(triag_hex_index_to_row_col(2, 2))
    print(triag_hex_index_to_row_col(3, 2))
    print(triag_hex_index_to_row_col(4, 2))
    print(triag_hex_index_to_row_col(5, 2))
    print(triag_hex_index_to_row_col(6, 2))
    print(triag_hex_index_to_row_col(7, 2))
    print(triag_hex_index_to_row_col(8, 2))

    print(triag_hex_row_col_to_index(0, 0, 2))
    print(triag_hex_row_col_to_index(1, 0, 2))
    print(triag_hex_row_col_to_index(1, 1, 2))
    print(triag_hex_row_col_to_index(2, 0, 2))
    print(triag_hex_row_col_to_index(2, 1, 2))
    print(triag_hex_row_col_to_index(2, 2, 2))
    print(triag_hex_row_col_to_index(3, 0, 2))
    print(triag_hex_row_col_to_index(3, 1, 2))
    print(triag_hex_row_col_to_index(4, 0, 2))
