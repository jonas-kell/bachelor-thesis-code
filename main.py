import jax
from jax.config import config

# 64 bit processing
config.update("jax_enable_x64", True)

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
from models.preconfigured import cnn


if __name__ == "__main__":
    n_steps = 1000
    n_samples = 40000

    lattice_shape = "linear"
    # lattice_shape = "cubic"
    # lattice_shape = "trigonal_square"
    # lattice_shape = "trigonal_diamond"
    # lattice_shape = "trigonal_hexagonal"
    # lattice_shape = "hexagonal"

    lattice_size = 6
    lattice_periodic = True

    lattice_parameters = resolve_lattice_parameters(
        shape=lattice_shape, size=lattice_size, periodic=lattice_periodic
    )

    tensorboard_folder_path = "/media/jonas/69B577D0C4C25263/MLData/tensorboard_jax/"

    execute_ground_state_search(
        n_steps=n_steps,
        n_samples=n_samples,
        lattice_parameters=lattice_parameters,
        model_name="CNN",
        model=cnn(lattice_parameters["nr_sites"]),
        tensorboard_folder_path=tensorboard_folder_path,
        hamiltonian_J_parameter=-1.0,
        hamiltonian_h_parameter=-0.7,
    )
