import jax.numpy as jnp
import numpy as np
from typing import Literal


def transform_jax_zero_matrix_to_neg_infinity(matrix: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(matrix == 0.0, -jnp.inf, matrix)


# !! if the matrix has an empty row (unconnected node) in "avg" mode it throws an nan/inf error because of division by 0. Better use "avg+1" to be save
def get_jax_adjacency_matrix(
    list_representation: list,
    type: Literal["sum", "avg", "avg+1"] = "sum",
) -> jnp.ndarray:
    matrix = adjacency_matrix_from_list_representation(list_representation)

    if type == "sum":
        result = matrix
    if type == "avg":
        result = get_averaging_matrix(matrix, count_own_connection=False)
    if type == "avg+1":
        result = get_averaging_matrix(matrix, count_own_connection=True)

    return jnp.array(result, dtype=np.float32)


# !! if the matrix has an empty row (unconnected node) in "avg" mode it throws an nan/inf error because of division by 0. Better use "avg+1" to be save
def get_averaging_matrix(matrix: np.ndarray, count_own_connection=True) -> np.ndarray:
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert matrix.dtype == np.bool8

    n = matrix.shape[0]

    # init the matrix that counts the connections
    connections_count = np.zeros_like(matrix, dtype=np.float16)

    # for each connecting node, increase the count
    for i in range(n):
        for j in range(n):
            if i != j:
                connections_count[i, i] += matrix[i, j]
            elif count_own_connection:
                # add one if counting own connections
                connections_count[i, i] += 1

    # invert the matrix to be able to get  D -> D^(-1/2)
    # D^(-1/2) this notion breaks my brain, but is applicable here because the matrices are strict positive, diagonal and square
    for i in range(n):  # np functions do not play nice with my zeros -> loop
        connections_count[i, i] = connections_count[i, i] ** (-1 / 2)

    return connections_count @ matrix.astype(np.float16) @ connections_count


def adjacency_matrix_from_list_representation(list_representation: list) -> np.ndarray:
    assert type(list_representation) == list

    nr_sites = len(list_representation)

    result = np.zeros((nr_sites, nr_sites), dtype=np.bool8)

    for i, interaction_list in enumerate(list_representation):  # row index
        assert type(interaction_list) == list

        for j in interaction_list:  # column index
            result[
                i,
                j,
            ] = 1

    return result


if __name__ == "__main__":
    test_list = list([[1], [0, 3], [3], [0]])

    print(test_list)
    # print(adjacency_matrix_from_list_representation(test_list))

    print(get_jax_adjacency_matrix(test_list, "avg+1"))

    print(
        transform_jax_zero_matrix_to_neg_infinity(
            get_jax_adjacency_matrix(test_list, "avg+1")
        )
    )
