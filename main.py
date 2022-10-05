import os

disable_preallocation = True
if disable_preallocation:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    print("CAUTION. Jax preallocation is disabled. This may hurt performance")

import jax
from jax.config import config

# 64 bit processing
config.update("jax_enable_x64", True)

disable_jit = False
config.update("jax_disable_jit", disable_jit)
if disable_jit:
    print("CAUTION. JIT-Compilation is deactivated for debugging purposes")

debug_nans = False
if debug_nans:
    print("CAUTION. Debug NANs is activated for debugging purposes")
config.update("jax_debug_nans", debug_nans)

# Check whether GPU is available
gpu_avail = jax.lib.xla_bridge.get_backend().platform == "gpu"

if gpu_avail:
    print("Running on GPU")

if not gpu_avail:
    print("Running on CPU not supported. Too slow and therefore not easily comparable")

import sys
from typing import Literal

sys.path.append(os.path.abspath("./structures"))
from structures.lattice_parameter_resolver import resolve_lattice_parameters

sys.path.append(os.path.abspath("./computation"))
from computation.ground_state_search import execute_ground_state_search

sys.path.append(os.path.abspath("./models"))
from models.preconfigured import cnn, rbm
from models.metaformer import (
    transformer,
    graph_transformer_nn,
    graph_transformer_nnn,
    graph_poolformer_nn,
    graph_poolformer_nnn,
    graph_conformer_nn,
    graph_conformer_nnn,
)

# local folder constants
tensorboard_folder_path = "/media/jonas/69B577D0C4C25263/MLData/tensorboard/"


# add custom configurations in this dict
available_models = {
    "CNN": lambda lattice_parameters, depth, embed_dim, num_heads, mlp_ratio, ansatz: cnn(  # ignore graph specific params
        lattice_parameters["nr_sites"], ansatz
    ),
    "RBM": lambda lattice_parameters, depth, embed_dim, num_heads, mlp_ratio, ansatz: rbm(
        ansatz
    ),  # ignore graph specific params
    "TF": lambda lattice_parameters, depth, embed_dim, num_heads, mlp_ratio, ansatz: transformer(
        lattice_parameters=lattice_parameters,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        ansatz=ansatz,
    ),
    "GTF-NN": lambda lattice_parameters, depth, embed_dim, num_heads, mlp_ratio, ansatz: graph_transformer_nn(
        lattice_parameters=lattice_parameters,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        ansatz=ansatz,
    ),
    "GTF-NNN": lambda lattice_parameters, depth, embed_dim, num_heads, mlp_ratio, ansatz: graph_transformer_nnn(
        lattice_parameters=lattice_parameters,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        ansatz=ansatz,
    ),
    # TODO "dumb" pooling action as comparison
    "GPF-NN": lambda lattice_parameters, depth, embed_dim, num_heads, mlp_ratio, ansatz: graph_poolformer_nn(
        lattice_parameters=lattice_parameters,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        ansatz=ansatz,
    ),
    "GPF-NNN": lambda lattice_parameters, depth, embed_dim, num_heads, mlp_ratio, ansatz: graph_poolformer_nnn(
        lattice_parameters=lattice_parameters,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        ansatz=ansatz,
    ),
    "SGDCF-NN": lambda lattice_parameters, depth, embed_dim, num_heads, mlp_ratio, ansatz: graph_conformer_nn(
        lattice_parameters=lattice_parameters,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        ansatz=ansatz,
    ),
    "SGDCF-NNN": lambda lattice_parameters, depth, embed_dim, num_heads, mlp_ratio, ansatz: graph_conformer_nnn(
        lattice_parameters=lattice_parameters,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        ansatz=ansatz,
    ),
}


def execute_computation(
    n_steps: int,
    n_samples: int,
    lattice_shape: Literal[
        "linear",
        "square",
        "trigonal_square",
        "trigonal_diamond",
        "trigonal_hexagonal",
        "hexagonal",
    ],
    lattice_size: int,
    lattice_periodic: bool,
    lattice_random_swaps: int,
    model_name: str,
    hamiltonian_J_parameter: float = -1.0,
    hamiltonian_h_parameter: float = -0.7,
    num_chains: int = 100,
    thermalization_sweeps: int = 25,
    nqs_batch_size: int = 1000,
    depth: int = 5,
    embed_dim: int = 6,
    num_heads: int = 3,
    mlp_ratio: int = 2,
    ansatz: Literal[
        "single-real", "single-complex", "split-complex", "two-real"
    ] = "single-real",
    head: Literal["act-fun", "cnn"] = "act-fun",
    early_abort_var: float = -1.0,
):
    lattice_parameters = resolve_lattice_parameters(
        shape=lattice_shape,
        size=lattice_size,
        periodic=lattice_periodic,
        random_swaps=lattice_random_swaps,
    )

    if model_name in available_models:
        model_fn = available_models[model_name]

    execute_ground_state_search(
        n_steps=n_steps,
        n_samples=n_samples,
        lattice_parameters=lattice_parameters,
        model_name=model_name,
        model_fn=model_fn,
        tensorboard_folder_path=tensorboard_folder_path,
        hamiltonian_J_parameter=hamiltonian_J_parameter,
        hamiltonian_h_parameter=hamiltonian_h_parameter,
        num_chains=num_chains,
        thermalization_sweeps=thermalization_sweeps,
        nqs_batch_size=nqs_batch_size,
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        ansatz=ansatz,
        head=head,
        early_abort_var=early_abort_var,
    )


if __name__ == "__main__":
    parameters = {
        "n_steps": 1000,
        "n_samples": 1000,
        "lattice_shape": "trigonal_square",
        "lattice_size": 4,
        "lattice_periodic": True,
        "lattice_random_swaps": 0,
        "model_name": "SGDCF-NNN",
        "hamiltonian_J_parameter": -1.0,
        "hamiltonian_h_parameter": -0.7,
        "num_chains": 100,
        "thermalization_sweeps": 25,
        "nqs_batch_size": 1000,
        "depth": 3,
        "embed_dim": 8,
        "num_heads": 2,
        "mlp_ratio": 4,
        "ansatz": "single-real",
        "head": "act-fun",
        "early_abort_var": -1.0,
    }

    additional_parameter_strings = [] if len(sys.argv) < 2 else sys.argv[1:]

    for additional_parameter_string in additional_parameter_strings:
        split = additional_parameter_string.split("=", 1)

        if len(split) == 2 and split[0] in parameters:
            parameters[split[0]] = type(parameters[split[0]])(split[1])
        else:
            print(f"Unknown parameter+value: {split}")

    for param in parameters:
        print(f"Using parameter   {param:26} with value:    {parameters[param]}")

    execute_computation(**parameters)
