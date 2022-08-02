import jax
from jax.config import config

config.update("jax_enable_x64", True)

import os
import sys

import jax.random as random

sys.path.append(os.path.abspath("./../"))
sys.path.append(os.path.abspath("./../models"))
sys.path.append(os.path.abspath("./../structures"))
from models.metaformer import (
    graph_transformer_nnn,
    graph_conformer_nnn,
    graph_poolformer_nnn,
)
from structures.lattice_parameter_resolver import resolve_lattice_parameters


input_size = 25

lattice_parameters = resolve_lattice_parameters(
    shape="linear", size=input_size, periodic=False
)

model = graph_poolformer_nnn(
    lattice_parameters=lattice_parameters,
    depth=10,
    embed_dim=6,
    num_heads=3,
    mlp_ratio=2,
)

x = random.randint(random.PRNGKey(0), (input_size,), 0, 2)

params = model.init(random.PRNGKey(0), x)

# print(jax.tree_util.tree_map(lambda x: x.shape, params))
number_model_parameters = sum(x.size for x in jax.tree_leaves(params))
print(f"The model has {number_model_parameters} parameters")

model_apply = jax.jit(lambda p, x: model.apply(p, x))

for i in range(1):
    x = random.randint(random.PRNGKey(0), (input_size,), 0, 2)
    print(model_apply(params, x))
