import jax
from jax.config import config

config.update("jax_enable_x64", True)

import os
import sys

import jax.numpy as jnp
import jax.random as random

sys.path.append(os.path.abspath("./../"))
from models.metaformer import metaformer_base
from models.preconfigured import cnn
from structures.lattice_parameter_resolver import resolve_lattice_parameters


input_size = 15

lattice_parameters = resolve_lattice_parameters(
    shape="linear", size=input_size, periodic=True
)

model = metaformer_base(lattice_parameters=lattice_parameters)
# model = cnn(input_size)

x = jnp.ones(input_size)

params = model.init(random.PRNGKey(0), x)

print(params)

print(model.apply(params, x))
