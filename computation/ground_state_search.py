"""
Mostly copy-paste from jvmc example initially. Modified to fit my needs
https://github.com/markusschmitt/vmc_jax/blob/master/examples/ex0_ground_state_search.py
"""

from typing import Callable, Literal
import jax
import jVMC
import jax.random as random
import flax.linen as nn
from jax.config import config

config.update("jax_enable_x64", True)

import datetime
import os
import sys
import time
import numpy as np

sys.path.append(os.path.abspath("./../structures"))

from torch.utils.tensorboard import SummaryWriter
from structures.lattice_parameter_resolver import LatticeParameters
from hamiltonian import get_hamiltonian


def execute_ground_state_search(
    n_steps: int,
    n_samples: int,
    lattice_parameters: LatticeParameters,
    model_name: str,
    model_fn: Callable[
        [
            LatticeParameters,
            int,  # depth
            int,  # embed_dim
            int,  # num_heads
            int,  # mlp_ratio
            Literal["single-real", "single-complex", "single-split", "two-real"],
        ],
        nn.Module,
    ],
    tensorboard_folder_path: str,
    hamiltonian_J_parameter: int = -1,
    hamiltonian_h_parameter: int = -0.7,
    num_chains: int = 100,
    thermalization_sweeps: int = 25,
    nqs_batch_size: int = 1000,
    depth: int = 5,
    embed_dim: int = 6,
    num_heads: int = 3,
    mlp_ratio: int = 2,
    ansatz: Literal[
        "single-real", "single-complex", "single-split", "two-real"
    ] = "single-real",
    early_abort_var: float = -1.0,
):
    # Get lattice parameters
    L = lattice_parameters["nr_sites"]

    print(
        f"We calculate the '{lattice_parameters['shape_name']}' lattice of size {lattice_parameters['size']} which means it has {L} lattice-sites. The boundary condition is {'periodic' if lattice_parameters['periodic'] else 'Non-periodic'}"
    )

    # get model
    model = model_fn(lattice_parameters, depth, embed_dim, num_heads, mlp_ratio, ansatz)

    # Variational wave function
    psi = jVMC.vqs.NQS(model, seed=1234, batchSize=nqs_batch_size)

    # Set up hamiltonian
    hamiltonian = get_hamiltonian(
        lattice_parameters=lattice_parameters,
        hamiltonian_J_parameter=hamiltonian_J_parameter,
        hamiltonian_h_parameter=hamiltonian_h_parameter,
    )

    # Set up sampler
    sampler = jVMC.sampler.MCSampler(
        psi,
        (L,),
        random.PRNGKey(4321),
        updateProposer=jVMC.sampler.propose_spin_flip_Z2,
        numChains=num_chains,
        sweepSteps=L,
        numSamples=n_samples,
        thermalizationSweeps=thermalization_sweeps,
    )

    # Number of parameters
    number_model_parameters = sum(x.size for x in jax.tree_util.tree_leaves(psi.params))
    print(f"The model has {number_model_parameters} parameters")

    # Set up TDVP
    tdvpEquation = jVMC.util.tdvp.TDVP(
        sampler, rhsPrefactor=1.0, svdTol=1e-8, diagonalShift=10, makeReal="real"
    )

    # ODE integrator
    stepper = jVMC.util.stepper.Euler(timeStep=1e-2)

    # Setup tensorboard Summary Writer
    flush_secs = 2

    run_date = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    run_name = (
        lattice_parameters["shape_name"]
        + f"({lattice_parameters['size']},{lattice_parameters['nr_sites']},{'p' if lattice_parameters['periodic']else 'np'},{lattice_parameters['nr_random_swaps']})"
        + "_with_"
        + model_name
        + "_at_"
        + run_date
    )
    tensorboard_folder = os.path.join(tensorboard_folder_path, run_name)

    writer = SummaryWriter(
        tensorboard_folder,
        flush_secs=flush_secs,
    )
    hparams_to_log = {
        "model_name": model_name,
        "number_model_parameters": number_model_parameters,
        "depth": depth,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "mlp_ratio": mlp_ratio,
        "ansatz": ansatz,
        "lattice_shape_name": lattice_parameters["shape_name"],
        "lattice_nr_sites": lattice_parameters["nr_sites"],
        "lattice_periodic": lattice_parameters["periodic"],
        "lattice_size": lattice_parameters["size"],
        "lattice_nr_random_swaps": lattice_parameters["nr_random_swaps"],
        "hamiltonian_J_parameter": hamiltonian_J_parameter,
        "hamiltonian_h_parameter": hamiltonian_h_parameter,
        "n_samples": n_samples,
        "n_steps": n_steps,
        "num_chains": num_chains,
        "thermalization_sweeps": thermalization_sweeps,
        "nqs_batch_size": nqs_batch_size,
    }
    # (threshold, reached)
    var_thresholds = [
        [0.2, False],
        [0.05, False],
        [0.02, False],
        [0.01, False],
        [0.005, False],
        [0.002, False],
    ]
    hparams_logger_vals = {
        "var_below_0.2": n_steps,
        "var_below_0.05": n_steps,
        "var_below_0.02": n_steps,
        "var_below_0.01": n_steps,
        "var_below_0.005": n_steps,
        "var_below_0.002": n_steps,
    }
    # the hyperparameter module is used normally to compare used hyperparamaters for multiple runs in one "script-execution". I can only get information about the interesting metrics (loss/accuracy/...) after my model has trained sufficiently long. It may crash or be aborted earlier however, what would result in not writing the corresponding hyperparameter entry. Therefor the hyperparameters are added everytime there is an update. Only the latest written info will be used by tensorboard
    writer.add_hparams(
        hparams_to_log,
        hparams_logger_vals,
        run_name="./",
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
        # print(psi.params) # Debug if network parameters get backpropagated
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

        # write var milestones for hparam tracking
        var = float(tdvpEquation.ElocVar0 / L)
        if n > 4:  # small warmup period
            for threshold_array in var_thresholds:
                if var < threshold_array[0] and not threshold_array[1]:
                    threshold_array[1] = True
                    hparams_logger_vals[f"var_below_{str(threshold_array[0])}"] = n
                    writer.add_hparams(
                        hparams_to_log,
                        hparams_logger_vals,
                        run_name="./",
                    )

            # abort early when target performance is reached
            if early_abort_var > 0 and var < early_abort_var:
                break

    # close the writer
    writer.close()
