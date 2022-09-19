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
from models.metaformer import Attention, AveragingPoolingHead
from structures.lattice_parameter_resolver import resolve_lattice_parameters


embed_dim = 8 * 2
nr_patches = 15

lattice_parameters = resolve_lattice_parameters(
    shape="linear", size=nr_patches, periodic=True
)


# Test Attention

x = jnp.ones((nr_patches, embed_dim))
print("x shape: ", x.shape)
model = Attention(
    lattice_parameters=lattice_parameters,
    embed_dim=embed_dim,
    num_heads=8,
    qkv_bias=True,
    mixing_symmetry="symm_nnn",
)
params = model.init(random.PRNGKey(0), x)
print(model.apply(params, x).shape)


# Test Averaging Convolution Head

x = jnp.array(range(nr_patches * embed_dim)).reshape((nr_patches, embed_dim))
print("x: ", x)
print("x shape: ", x.shape)

model = AveragingPoolingHead()
params = model.init(random.PRNGKey(0), x)

print(model.apply(params, x))
print(model.apply(params, x).shape)
