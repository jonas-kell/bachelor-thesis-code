def square_lattice_get_nn_indices(index, n, periodic_bounds=False):
    assert n > 0
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
    assert n > 0
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
    assert n > 0
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
    assert n > 0
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


def triangular_diamond_lattice_get_nn_indices(index, n, periodic_bounds=False):
    assert n > 0

    neighbors = []

    if not 0 <= index < (n + 1) * (n + 1):
        return neighbors

    y, x = triangular_dimond_index_to_row_col(index, n)

    for xi, yi in [
        (x + (0 if y <= n else 1), y - 1),
        (x + (0 if y <= n else 1) - 1, y - 1),
        (x - 1, y),
        (x + 1, y),
        (x + (0 if y < n else -1), y + 1),
        (x + (0 if y < n else -1) + 1, y + 1),
    ]:
        neighbors.append(
            triangular_dimond_row_col_to_index(
                yi, xi, n, periodic_bounds=periodic_bounds
            )
        )

    # filter neg values (may be generated by `triag_dimond_row_col_to_index` if periodic_bounds=False)
    neighbors = [i for i in neighbors if i >= 0]

    return neighbors


def triangular_diamond_lattice_get_nnn_indices(index, n, periodic_bounds=False):
    assert n > 0

    neighbors = []

    if not 0 <= index < (n + 1) * (n + 1):
        return neighbors

    y, x = triangular_dimond_index_to_row_col(index, n)

    for xi, yi in [
        (x + (-1 if y <= n else (0 if y == n + 1 else 1)), y - 2),
        (x + (-1 if y >= n else (0 if y == n - 1 else 1)), y + 2),
        (x + (0 if y <= n else 1) + 1, y - 1),
        (x + (0 if y <= n else 1) - 2, y - 1),
        (x + (0 if y < n else -1) - 1, y + 1),
        (x + (0 if y < n else -1) + 2, y + 1),
    ]:
        neighbors.append(
            triangular_dimond_row_col_to_index(
                yi, xi, n, periodic_bounds=periodic_bounds
            )
        )

    # filter neg values (may be generated by `triag_dimond_row_col_to_index` if periodic_bounds=False)
    neighbors = [i for i in neighbors if i >= 0]

    return neighbors


def triangular_dimond_index_to_row_col(index, n):
    assert n > 0

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


def triangular_dimond_row_col_to_index(row, col, n, periodic_bounds=True):
    assert n > 0

    if (
        (row < 0)
        or (col < 0)
        or (row > 2 * n)
        or (row <= n and col > row)
        or (row > n and col > 2 * n - row)
    ):
        if periodic_bounds:
            # transform row/col into valid space
            # as this coordinate space is trash, transform into axis-cartesian coordinates and then transfrom back

            # print("before", row, col)
            cart_row = col + max(row - n, 0)
            cart_col = col + max(n - row, 0)

            # print("cart", cart_row, cart_col)
            cart_row = cart_row % (n + 1)
            cart_col = cart_col % (n + 1)

            row = n - cart_col + cart_row
            col = min(cart_col, cart_row)
            # print("after", row, col)
        else:
            # return invalid
            return -1

    index = 0
    for i in [n - abs(j) + 1 for j in range(-n, row - n)]:
        index += i

    index += col

    return index


def triangular_hexagonal_lattice_get_nn_indices(index, n, periodic_bounds=False):
    assert n > 1
    neighbors = []

    if not 0 <= index < triangular_hexagonal_nr_lattice_sites(n):
        return neighbors

    q, r = triangular_hexagonal_index_to_qr(index, n)

    for qi, ri in [
        (q, r - 1),
        (q + 1, r - 1),
        (q + 1, r),
        (q, r + 1),
        (q - 1, r + 1),
        (q - 1, r),
    ]:
        neighbors.append(
            triangular_hexagonal_qr_to_index(qi, ri, n, periodic_bounds=periodic_bounds)
        )

    # filter neg values (may be generated by `triangular_hexagonal_qr_to_index` if periodic_bounds=False)
    neighbors = [i for i in neighbors if i >= 0]

    return neighbors


def triangular_hexagonal_lattice_get_nnn_indices(index, n, periodic_bounds=False):
    assert n > 1
    neighbors = []

    if not 0 <= index < triangular_hexagonal_nr_lattice_sites(n):
        return neighbors

    q, r = triangular_hexagonal_index_to_qr(index, n)

    for qi, ri in [
        (q + 1, r - 2),
        (q + 2, r - 1),
        (q + 1, r + 1),
        (q - 1, r + 2),
        (q - 2, r + 1),
        (q - 1, r - 1),
    ]:
        neighbors.append(
            triangular_hexagonal_qr_to_index(qi, ri, n, periodic_bounds=periodic_bounds)
        )

    # filter neg values (may be generated by `triangular_hexagonal_qr_to_index` if periodic_bounds=False)
    neighbors = [i for i in neighbors if i >= 0]

    return neighbors


def triangular_hexagonal_index_to_qr(index, n):
    assert n > 1
    assert 0 <= index < triangular_hexagonal_nr_lattice_sites(n)

    index_test = 0
    for r in range(-n + 1, n):
        for q in range(-n + 1, n):
            s = 0 - q - r

            if max(abs(q), abs(r), abs(s)) < n:
                if index_test == index:
                    return q, r
                index_test += 1

    return -1, -1


def triangular_hexagonal_qr_to_index(q, r, n, periodic_bounds=True):
    assert n > 1

    s = 0 - q - r
    if max(abs(q), abs(r), abs(s)) >= n:
        if periodic_bounds:
            fixed = False
            for try_q, try_r in [
                (q - 2 * n + 1, r + n - 1),
                (q - n + 1, r - n),
                (q + n, r - 2 * n + 1),
                (q + 2 * n - 1, r - n + 1),
                (q + n - 1, r + n),
                (q - n, r + 2 * n - 1),
            ]:
                try_s = 0 - try_q - try_r

                if max(abs(try_q), abs(try_r), abs(try_s)) < n:
                    q = try_q
                    r = try_r
                    s = try_s

                    fixed = True
                    break

            if not fixed:
                raise RuntimeError(
                    f"Too far into periodic bounds, not defined: q({q}) r({r}) s({s}),  n({n})"
                )
        else:
            return -1

    index = 0
    for r_test in range(-n + 1, n):
        for q_test in range(-n + 1, n):
            s_test = 0 - q_test - r_test

            if max(abs(q_test), abs(r_test), abs(s_test)) < n:
                if q == q_test and r == r_test:
                    return index

                index += 1
    return -1


# too lazy to calculate closed expression for this. Sry
def triangular_hexagonal_nr_lattice_sites(n):
    assert n > 1

    number = 0

    for r in range(-n + 1, n):
        for q in range(-n + 1, n):
            s = 0 - q - r

            if max(abs(q), abs(r), abs(s)) < n:
                number += 1

    return number


def hexagonal_index_to_qr(index, n):
    assert n > 0
    assert 0 <= index < hexagonal_nr_lattice_sites(n)

    index_test = 0
    start_q, start_r = (n - 1, -2 * (n - 1) - 1)

    for (q_step, r_step), steps in hex_step_iterator(n):
        for i in range(steps):
            q = start_q + i
            r = start_r

            if qr_part_of_hexagonal_lattice(q, r):
                if index_test == index:
                    return q, r
                index_test += 1

        start_q += q_step
        start_r += r_step

    return -1, -1


def hexagonal_qr_to_index(q, r, n, periodic_bounds=True):
    assert n > 0

    for try_q, try_r in [
        (q, r),
        (q + 3 * n, r - 3 * n),
        (q - 3 * n, r + 3 * n),
        (q, r - 3 * n),
        (q, r + 3 * n),
        (q + 3 * n, r),
        (q - 3 * n, r),
    ]:
        # try translation to index for each possible shift (first is unshifted)
        index = 0
        start_q, start_r = (n - 1, -2 * (n - 1) - 1)

        for (q_step, r_step), steps in hex_step_iterator(n):
            for i in range(steps):
                q_test = start_q + i
                r_test = start_r

                if qr_part_of_hexagonal_lattice(q_test, r_test):
                    if try_q == q_test and try_r == r_test:
                        return index  # break out of everything and return
                    index += 1

            start_q += q_step
            start_r += r_step

        # unshifted coords could not be translated, ok for non-periodic lattice
        if not periodic_bounds:
            return -1

    # no shift could translate coords, equals error
    raise RuntimeError(
        f"Too far into periodic bounds or non-index translatable, not defined: q({q}) r({r}),  n({n})"
    )


def hex_step_iterator(n):
    assert n > 0

    jumps_to_next_start = (
        [(-2, +1)] * (n - 1) + [(-1, +1), (0, +1)] * n + [(+1, +1)] * (n - 1) + [(0, 0)]
    )
    steps_to_take = (
        [2 * i + i - 1 for i in range(1, n)]
        + [3 * n - 1, 3 * n] * n
        + [2 * i + i - 1 for i in range(n, 0, -1)]
    )

    return zip(jumps_to_next_start, steps_to_take)


def hexagonal_lattice_get_nn_indices(index, n, periodic_bounds=False):
    assert n > 0
    neighbors = []

    if not 0 <= index < hexagonal_nr_lattice_sites(n):
        return neighbors

    q, r = hexagonal_index_to_qr(index, n)

    for qi, ri in [
        (q, r - 1),
        (q + 1, r - 1),
        (q + 1, r),
        (q, r + 1),
        (q - 1, r + 1),
        (q - 1, r),
    ]:
        if qr_part_of_hexagonal_lattice(qi, ri):
            neighbors.append(
                hexagonal_qr_to_index(qi, ri, n, periodic_bounds=periodic_bounds)
            )

    # filter neg values (may be generated by `triangular_hexagonal_qr_to_index` if periodic_bounds=False)
    neighbors = [i for i in neighbors if i >= 0]

    return neighbors


def hexagonal_lattice_get_nnn_indices(index, n, periodic_bounds=False):
    assert n > 0
    neighbors = []

    if not 0 <= index < hexagonal_nr_lattice_sites(n):
        return neighbors

    q, r = hexagonal_index_to_qr(index, n)

    for qi, ri in [
        (q + 1, r - 2),
        (q + 2, r - 1),
        (q + 1, r + 1),
        (q - 1, r + 2),
        (q - 2, r + 1),
        (q - 1, r - 1),
    ]:
        if qr_part_of_hexagonal_lattice(qi, ri):
            neighbors.append(
                hexagonal_qr_to_index(qi, ri, n, periodic_bounds=periodic_bounds)
            )

    # filter neg values (may be generated by `triangular_hexagonal_qr_to_index` if periodic_bounds=False)
    neighbors = [i for i in neighbors if i >= 0]

    return neighbors


def hexagonal_nr_lattice_sites(n):
    assert n > 0

    return 6 * n**2


def qr_part_of_hexagonal_lattice(q, r):
    nr_up_shifts = r // 2
    r = r - nr_up_shifts * 2
    q = q + nr_up_shifts

    q = q % 3

    return not ((q == 0 and r == 0) or (q == 1 and r == 1))


if __name__ == "__main__":
    # print(square_lattice_get_nn_indices(4, 3))
    # print(square_lattice_get_nn_indices(0, 3))

    # print(triangular_dimond_index_to_row_col(0, 2))
    # print(triangular_dimond_index_to_row_col(1, 2))
    # print(triangular_dimond_index_to_row_col(2, 2))
    # print(triangular_dimond_index_to_row_col(3, 2))
    # print(triangular_dimond_index_to_row_col(4, 2))
    # print(triangular_dimond_index_to_row_col(5, 2))
    # print(triangular_dimond_index_to_row_col(6, 2))
    # print(triangular_dimond_index_to_row_col(7, 2))
    # print(triangular_dimond_index_to_row_col(8, 2))

    # print(triangular_dimond_row_col_to_index(0, 0, 2))
    # print(triangular_dimond_row_col_to_index(1, 0, 2))
    # print(triangular_dimond_row_col_to_index(1, 1, 2))
    # print(triangular_dimond_row_col_to_index(2, 0, 2))
    # print(triangular_dimond_row_col_to_index(2, 1, 2))
    # print(triangular_dimond_row_col_to_index(2, 2, 2))
    # print(triangular_dimond_row_col_to_index(3, 0, 2))
    # print(triangular_dimond_row_col_to_index(3, 1, 2))
    # print(triangular_dimond_row_col_to_index(4, 0, 2))

    # print(qr_part_of_hexagonal_lattice(0, 0))
    # print(qr_part_of_hexagonal_lattice(3, 0))
    # print(qr_part_of_hexagonal_lattice(2, 2))
    # print(qr_part_of_hexagonal_lattice(-5, 1))
    # print(qr_part_of_hexagonal_lattice(2, -1))
    # print("asd")
    # print(qr_part_of_hexagonal_lattice(0, -1))
    # print(qr_part_of_hexagonal_lattice(1, -1))
    # print(qr_part_of_hexagonal_lattice(1, 0))
    # print(qr_part_of_hexagonal_lattice(0, 1))
    # print(qr_part_of_hexagonal_lattice(-1, 1))
    # print(qr_part_of_hexagonal_lattice(-1, 1))
    # print(qr_part_of_hexagonal_lattice(-1, 0))

    pass
