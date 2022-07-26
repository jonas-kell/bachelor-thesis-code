"""
Mostly copy-paste from jvmc example initially. Modified to fit my needs
https://github.com/markusschmitt/vmc_jax/blob/master/examples/ex0_ground_state_search.py
"""

import jax
from jax.config import config

# 64 bit processing
config.update("jax_enable_x64", True)

import jax.random as random
import jax.numpy as jnp
import flax.linen as nn

import jVMC

import datetime
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath("./structures"))

from torch.utils.tensorboard import SummaryWriter

from structures.lattice_parameter_resolver import resolve_lattice_parameters

# Check whether GPU is available
gpu_avail = jax.lib.xla_bridge.get_backend().platform == "gpu"

if gpu_avail:
    print("Running on GPU")

if not gpu_avail:
    print("Running on CPU not supported")
    exit()

# Get lattice parameters

lattice_shape = "linear"
# lattice_shape = "cubic"
# lattice_shape = "trigonal_square"
lattice_shape = "trigonal_diamond"
# lattice_shape = "trigonal_hexagonal"
# lattice_shape = "hexagonal"

periodic = True
size = 6

(
    nr_lattice_sites,
    nn_interaction_indices,
    nnn_interaction_indices,
) = resolve_lattice_parameters(size, lattice_shape, periodic)

L = nr_lattice_sites
g = -0.7

print(
    f"We calculate the '{lattice_shape}' lattice of size {size} which means it has {L} lattice-sites. The boundary condition is {'periodic' if periodic else 'Non-periodic'}"
)

# Initialize net
if gpu_avail:

    def myActFun(x):
        return 1 + nn.elu(x)

    net = jVMC.nets.CNN(
        F=(L,), channels=(16,), strides=(1,), periodicBoundary=True, actFun=(myActFun,)
    )
    n_steps = 1000
    n_Samples = 40000


# Variational wave function
psi = jVMC.vqs.NQS(net, seed=1234)


# Set up hamiltonian
hamiltonian = jVMC.operator.BranchFreeOperator()

interaction_used = np.zeros((L, L), dtype=np.bool8)
for l in range(L):
    for l_n in nn_interaction_indices[l]:
        if not interaction_used[l, l_n]:
            interaction_used[l_n, l] = interaction_used[l, l_n] = True
            hamiltonian.add(
                jVMC.operator.scal_opstr(
                    -1,
                    (
                        jVMC.operator.Sz(l),
                        jVMC.operator.Sz(l_n),
                    ),
                )
            )
    hamiltonian.add(jVMC.operator.scal_opstr(g, (jVMC.operator.Sx(l),)))


# Set up sampler
sampler = jVMC.sampler.MCSampler(
    psi,
    (L,),
    random.PRNGKey(4321),
    updateProposer=jVMC.sampler.propose_spin_flip_Z2,
    numChains=100,
    sweepSteps=L,
    numSamples=n_Samples,
    thermalizationSweeps=25,
)

# Set up TDVP
tdvpEquation = jVMC.util.tdvp.TDVP(
    sampler, rhsPrefactor=1.0, svdTol=1e-8, diagonalShift=10, makeReal="real"
)

# ODE integrator
stepper = jVMC.util.stepper.Euler(timeStep=1e-2)

# Setup tensorboard Summary Writer
flush_secs = 2

tensorboard_folder_path = "/media/jonas/69B577D0C4C25263/MLData/tensorboard_jax/"
run_date = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
run_name = "RANDOM_MODEL" + "_at_" + run_date
tensorboard_folder = os.path.join(tensorboard_folder_path, run_name)

writer = SummaryWriter(
    tensorboard_folder,
    flush_secs=flush_secs,
)

# fitting loop
for n in range(n_steps):

    # time
    start_processing_time = time.time()

    # step
    dp, _ = stepper.step(
        0,
        tdvpEquation,
        psi.get_parameters(),
        hamiltonian=hamiltonian,
        psi=psi,
        numSamples=None,
    )
    psi.set_parameters(dp)

    # time
    processing_time = time.time() - start_processing_time

    # tensorboard logs
    print(
        f"Step [{n:>5d}/{n_steps:>5d}] E/L: {(jax.numpy.real(tdvpEquation.ElocMean0) / L):>6f} Var(E)/L: {(tdvpEquation.ElocVar0 / L):>6f} time: {processing_time:>2f}s"
    )
    writer.add_scalar("E/L", float(jax.numpy.real(tdvpEquation.ElocMean0) / L), n)
    writer.add_scalar("Var(E)/L", float(tdvpEquation.ElocVar0 / L), n)
    writer.add_scalar("time/s", float(processing_time), n)

# close the writer
writer.close()
