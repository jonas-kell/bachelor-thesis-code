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
    exit()

import sys

sys.path.append(os.path.abspath("./structures"))
from structures.lattice_parameter_resolver import resolve_lattice_parameters

sys.path.append(os.path.abspath("./computation"))
from computation.ground_state_search import execute_ground_state_search

sys.path.append(os.path.abspath("./models"))
from models.preconfigured import cnn, complexRBM
from models.metaformer import (
    transformer,
    graph_transformer_nn,
    graph_transformer_nnn,
)

# local folder constants
tensorboard_folder_path = "/media/jonas/69B577D0C4C25263/MLData/tensorboard_trash/"


# add custom configurations in this dict
available_models = {
    "CNN": lambda lattice_parameters: cnn(lattice_parameters["nr_sites"]),
    "RBM": lambda lattice_parameters: complexRBM(),
    "TF": lambda lattice_parameters: transformer(lattice_parameters=lattice_parameters),
    "GF-NN": lambda lattice_parameters: graph_transformer_nn(
        lattice_parameters=lattice_parameters
    ),
    "GF-NNN": lambda lattice_parameters: graph_transformer_nnn(
        lattice_parameters=lattice_parameters
    ),
}


def execute_computation(
    n_steps: int,
    n_samples: int,
    lattice_shape: str,
    lattice_size: int,
    lattice_periodic: bool,
    model_name: str,
    hamiltonian_J_parameter: float = -1.0,
    hamiltonian_h_parameter: float = -0.7,
    num_chains: int = 100,
    thermalization_sweeps: int = 25,
    nqs_batch_size: int = 1000,
):
    lattice_parameters = resolve_lattice_parameters(
        shape=lattice_shape, size=lattice_size, periodic=lattice_periodic
    )

    if model_name in available_models:
        model = available_models[model_name](lattice_parameters)

    execute_ground_state_search(
        n_steps=n_steps,
        n_samples=n_samples,
        lattice_parameters=lattice_parameters,
        model_name=model_name,
        model=model,
        tensorboard_folder_path=tensorboard_folder_path,
        hamiltonian_J_parameter=hamiltonian_J_parameter,
        hamiltonian_h_parameter=hamiltonian_h_parameter,
        num_chains=num_chains,
        thermalization_sweeps=thermalization_sweeps,
        nqs_batch_size=nqs_batch_size,
    )


# lattice_shape =

# "linear"
# "cubic"
# "trigonal_square"
# "trigonal_diamond"
# "trigonal_hexagonal"
# "hexagonal"


if __name__ == "__main__":
    parameters = {
        "n_steps": 1000,
        "n_samples": 40000,
        "lattice_shape": "linear",
        "lattice_size": 25,
        "lattice_periodic": True,
        "model_name": "CNN",
        "hamiltonian_J_parameter": -1.0,
        "hamiltonian_h_parameter": -0.7,
        "num_chains": 100,
        "thermalization_sweeps": 25,
        "nqs_batch_size": 1000,
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
