import imp
from neighbors import (
    linear_lattice_get_nn_indices,
    linear_lattice_get_nnn_indices,
    linear_nr_lattice_sites,
    cubic_lattice_get_nn_indices,
    cubic_lattice_get_nnn_indices,
    cubic_nr_lattice_sites,
    trigonal_square_lattice_get_nn_indices,
    trigonal_square_lattice_get_nnn_indices,
    trigonal_square_nr_lattice_sites,
    trigonal_diamond_lattice_get_nn_indices,
    trigonal_diamond_lattice_get_nnn_indices,
    trigonal_diamond_nr_lattice_sites,
    trigonal_hexagonal_lattice_get_nn_indices,
    trigonal_hexagonal_lattice_get_nnn_indices,
    trigonal_hexagonal_nr_lattice_sites,
    hexagonal_lattice_get_nn_indices,
    hexagonal_lattice_get_nnn_indices,
    hexagonal_nr_lattice_sites,
)
from typing import Literal, Tuple


def resolve_lattice_parameters(
    size: int,  # 1 -> ...
    lattice_shape: Literal[
        "linear",
        "cubic",
        "trigonal_square",
        "trigonal_diamond",
        "trigonal_hexagonal",
        "hexagonal",
    ],
    periodic: bool = False,
) -> Tuple[
    int, list, list
]:  # int: number_lattice_sites, list(list(int)): nn_interaction_indices, list(list(int)): nnn_interaction_indices
    assert int(size) == size
    assert size > 0

    if lattice_shape == "linear":
        n = size
        nr_function = linear_nr_lattice_sites
        nn_function = linear_lattice_get_nn_indices
        nnn_function = linear_lattice_get_nnn_indices

    if lattice_shape == "cubic":
        n = size + 1
        nr_function = cubic_nr_lattice_sites
        nn_function = cubic_lattice_get_nn_indices
        nnn_function = cubic_lattice_get_nnn_indices

    elif lattice_shape == "trigonal_square":
        n = 2 * size
        nr_function = trigonal_square_nr_lattice_sites
        nn_function = trigonal_square_lattice_get_nn_indices
        nnn_function = trigonal_square_lattice_get_nnn_indices

    elif lattice_shape == "trigonal_diamond":
        n = size
        nr_function = trigonal_diamond_nr_lattice_sites
        nn_function = trigonal_diamond_lattice_get_nn_indices
        nnn_function = trigonal_diamond_lattice_get_nnn_indices

    elif lattice_shape == "trigonal_hexagonal":
        n = size + 1
        nr_function = trigonal_hexagonal_nr_lattice_sites
        nn_function = trigonal_hexagonal_lattice_get_nn_indices
        nnn_function = trigonal_hexagonal_lattice_get_nnn_indices

    elif lattice_shape == "hexagonal":
        n = size
        nr_function = hexagonal_nr_lattice_sites
        nn_function = hexagonal_lattice_get_nn_indices
        nnn_function = hexagonal_lattice_get_nnn_indices

    else:
        raise RuntimeError("lattice_shape not implemented")

    # aggregate data
    nr_lattice_sites = nr_function(n)
    nn_interaction_indices = list(
        [nn_function(i, n, periodic_bounds=periodic) for i in range(nr_lattice_sites)]
    )
    nnn_interaction_indices = list(
        [nnn_function(i, n, periodic_bounds=periodic) for i in range(nr_lattice_sites)]
    )

    return (nr_lattice_sites, nn_interaction_indices, nnn_interaction_indices)


if __name__ == "__main__":
    print(resolve_lattice_parameters(1, "hexagonal", False))
