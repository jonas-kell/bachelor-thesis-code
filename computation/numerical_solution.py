import scipy
import jVMC
from operators import get_hamiltonian
import numpy as np
import time

from lattice_parameter_resolver import resolve_lattice_parameters


def get_index_from_z_eigenstate(eigenstate: list):
    index = 0

    for i, spin in enumerate(eigenstate):
        index += spin << i

    return index


def assemble_hamiltonian_in_eigenbasis(
    base_states: list,  # 1 x 2^L x L
    matrix_elements: list,  # 1 x 2^L x m
    configurations: list,  # 1 x 2^L x m x L
):
    assert len(base_states.shape) == 3
    assert len(matrix_elements.shape) == 3
    assert len(configurations.shape) == 4

    L = base_states.shape[2]
    m = configurations.shape[2]
    nr_states = 2**L

    assert base_states.shape[0] == 1
    assert base_states.shape[1] == nr_states
    assert base_states.shape[2] == L

    assert matrix_elements.shape[0] == 1
    assert matrix_elements.shape[1] == nr_states
    assert matrix_elements.shape[2] == m

    assert configurations.shape[0] == 1
    assert configurations.shape[1] == nr_states
    assert configurations.shape[2] == m
    assert configurations.shape[3] == L

    hamiltonian = scipy.sparse.dok_array((nr_states, nr_states), dtype=np.complex64)

    for i in range(nr_states):
        base_state_index = get_index_from_z_eigenstate(base_states[0, i])

        for j in range(m):
            hamiltonian[
                base_state_index, get_index_from_z_eigenstate(configurations[0, i, j])
            ] += matrix_elements[0, i, j]

    return hamiltonian


if __name__ == "__main__":
    start_time = time.time()

    lattice_parameters = resolve_lattice_parameters(
        size=6, shape="linear", periodic=True, random_swaps=0
    )
    L = lattice_parameters["nr_sites"]

    hamiltonian_J_parameter = -1.0
    hamiltonian_h_parameter = -0.7

    hamiltonian = get_hamiltonian(
        lattice_parameters=lattice_parameters,
        hamiltonian_J_parameter=hamiltonian_J_parameter,
        hamiltonian_h_parameter=hamiltonian_h_parameter,
    )

    sampler = jVMC.sampler.ExactSampler(lambda a: None, (L,))
    base_states = sampler.basis
    # print(base_states)                    # all obvious eigenstates of the spin-z-operator
    # print(base_states.shape)              # 1 x 2^L x L

    hamiltonian.get_s_primes(base_states)  # computes them and stores internally
    configurations = hamiltonian.sp
    matrix_elements = hamiltonian.matEl
    # print(configurations)                 # ALL eigenstate vectors that are a result of applying the Hamiltonian to one base_state
    # print(configurations.shape)           # 1 x 2^L x m x L (where m depends on the complexity of the Hamiltonian)
    # print(matrix_elements)                # the correspondung weight-factors to the result of applying the Hamiltonian to one base_state (correspond 1:1 with the "configurations" entries)
    # print(matrix_elements.shape)          # 1 x 2^L x m

    print(f"Creation of the hamiltonian took {time.time()-start_time:.3f}s")
    start_time = time.time()
    print(
        "Start to assemble the final matrix. This Code is very unoptimized, because its entirely written in python..."
    )

    eigenbasis_matrix = assemble_hamiltonian_in_eigenbasis(
        base_states, matrix_elements, configurations
    )

    print(
        f"The matrix conversion of the hamiltonian took {time.time()-start_time:.3f}s"
    )
    start_time = time.time()
    print("Matrix is assembled, start eigenvalue calculation... This may take long")

    eigenvalue = scipy.sparse.linalg.eigsh(
        A=eigenbasis_matrix,
        k=1,  # return the first eigenvalue.
        return_eigenvectors=False,
        sigma=-100,  # sigma "forces" that the smallest one is returend
    )

    print(f"Eigenvalue calculation took {time.time()-start_time:.3f}s")
    print(f"The Eigenvalue is {eigenvalue}, therefore E/L = {eigenvalue/L}")
