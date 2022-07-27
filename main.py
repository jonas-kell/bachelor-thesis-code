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


import os
import sys

sys.path.append(os.path.abspath("./structures"))
from structures.lattice_parameter_resolver import resolve_lattice_parameters

sys.path.append(os.path.abspath("./computation"))
from computation.ground_state_search import execute_ground_state_search

sys.path.append(os.path.abspath("./models"))
from models.preconfigured import cnn, complexRBM
from models.metaformer import Metaformer

# local folder constants
tensorboard_folder_path = "/media/jonas/69B577D0C4C25263/MLData/tensorboard_jax/"


# add custom configurations in this dict
available_models = {
    "CNN": lambda lattice_parameters: cnn(lattice_parameters["nr_sites"]),
    "RBM": lambda lattice_parameters: complexRBM(),
    "MB": lambda lattice_parameters: Metaformer(
        lattice_parameters=lattice_parameters,
        embed_dim=5,
        embed_mode="duplicate_nnn",
    ),
}


def execute_computation(
    n_steps: int,
    n_samples: int,
    lattice_shape: str,
    lattice_size: int,
    lattice_periodic: bool,
    model_name: str,
    hamiltonian_J_parameter=-1.0,
    hamiltonian_h_parameter=-0.7,
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
        "lattice_size": 15,
        "lattice_periodic": True,
        "model_name": "MB",
    }

    # TODO remove overwrite
    parameters["n_steps"] = 10
    parameters["n_samples"] = 50
    parameters["lattice_size"] = 16

    additional_parameter_strings = [] if len(sys.argv) < 2 else sys.argv[1:]

    for additional_parameter_string in additional_parameter_strings:
        split = additional_parameter_string.split("=", 1)

        if len(split) == 2 and split[0] in parameters:
            parameters[split[0]] = type(parameters[split[0]])(split[1])
        else:
            print(f"Unknown parameter+value: {split}")

    for param in parameters:
        print(f"Using parameter   {param:20} with value:    {parameters[param]}")

    execute_computation(**parameters)
