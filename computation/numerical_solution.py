import scipy
import jVMC
from hamiltonian import get_hamiltonian


from lattice_parameter_resolver import resolve_lattice_parameters

lattice_parameters = resolve_lattice_parameters(
    size=1, shape="cubic", periodic=True, random_swaps=0
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
base_states = sampler.basis[0:, 0:2, 0:]
print(base_states)
print(base_states.shape)

hamiltonian.get_s_primes(base_states)  # computes them and stores internally
configurations = hamiltonian.sp
matrix_elements = hamiltonian.matEl


print(configurations)
print(configurations.shape)
print(matrix_elements)
print(matrix_elements.shape)
