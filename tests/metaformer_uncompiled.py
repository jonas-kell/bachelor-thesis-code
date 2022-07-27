import jax
from jax.config import config

config.update("jax_enable_x64", True)

import os
import sys

import jax.numpy as jnp
import jax.random as random

sys.path.append(os.path.abspath("./../"))
from models.metaformer import Metaformer
from structures.lattice_parameter_resolver import resolve_lattice_parameters


input_size = 15

lattice_parameters = resolve_lattice_parameters(
    shape="linear", size=input_size, periodic=False
)

model = Metaformer(
    lattice_parameters=lattice_parameters, embed_dim=5, embed_mode="duplicate_spread"
)

x = random.randint(random.PRNGKey(0), (input_size,), 0, 2)

params = model.init(random.PRNGKey(0), x)

print(model.apply(params, x))
