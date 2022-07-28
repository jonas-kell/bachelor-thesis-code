import jax
from jax.config import config

config.update("jax_enable_x64", True)

import os
import sys

import jax.numpy as jnp
import jax.random as random

sys.path.append(os.path.abspath("./../"))
sys.path.append(os.path.abspath("./../models"))
sys.path.append(os.path.abspath("./../structures"))
from models.metaformer import Attention
from structures.lattice_parameter_resolver import resolve_lattice_parameters


embed_dim = 8 * 2
nr_patches = 15

lattice_parameters = resolve_lattice_parameters(
    shape="linear", size=nr_patches, periodic=False
)

model = Attention(
    embed_dim=embed_dim, num_heads=8, qkv_bias=True, mixing_symmetry="arbitrary"
)

x = jnp.ones((nr_patches, embed_dim))

params = model.init(random.PRNGKey(0), x)

print(model.apply(params, x).shape)
