import sys
import os

sys.path.append(os.path.abspath("../structures"))

from lattice_parameter_resolver import LatticeParameters

import numpy as np
import jVMC


def get_hamiltonian(
    lattice_parameters: LatticeParameters,
    hamiltonian_J_parameter,
    hamiltonian_h_parameter,
):
    L = lattice_parameters["nr_sites"]
    interaction_used = np.zeros((L, L), dtype=np.bool8)

    hamiltonian = jVMC.operator.BranchFreeOperator()
    for l in range(L):
        for l_n in lattice_parameters["nn_interaction_indices"][l]:
            if not interaction_used[l, l_n]:
                interaction_used[l_n, l] = interaction_used[l, l_n] = True
                hamiltonian.add(
                    jVMC.operator.scal_opstr(
                        hamiltonian_J_parameter,
                        (
                            jVMC.operator.Sz(l),
                            jVMC.operator.Sz(l_n),
                        ),
                    )
                )
        hamiltonian.add(
            jVMC.operator.scal_opstr(hamiltonian_h_parameter, (jVMC.operator.Sx(l),))
        )

    return hamiltonian
