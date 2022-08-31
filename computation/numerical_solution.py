import scipy
from hamiltonian import get_hamiltonian


from lattice_parameter_resolver import resolve_lattice_parameters

lattice_parameters = resolve_lattice_parameters(
    size=4, shape="cubic", periodic=True, random_swaps=0
)

hamiltonian_J_parameter = -1.0
hamiltonian_h_parameter = -0.7

hamiltonian = get_hamiltonian(
    lattice_parameters=lattice_parameters,
    hamiltonian_J_parameter=hamiltonian_J_parameter,
    hamiltonian_h_parameter=hamiltonian_h_parameter,
)

print(hamiltonian)
