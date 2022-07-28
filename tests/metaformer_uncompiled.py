import jax
from jax.config import config

config.update("jax_enable_x64", True)

import os
import sys

import jax.random as random

sys.path.append(os.path.abspath("./../"))
sys.path.append(os.path.abspath("./../models"))
sys.path.append(os.path.abspath("./../structures"))
from models.metaformer import graph_transformer_nnn
from structures.lattice_parameter_resolver import resolve_lattice_parameters


input_size = 25

lattice_parameters = resolve_lattice_parameters(
    shape="linear", size=input_size, periodic=False
)

model = graph_transformer_nnn(lattice_parameters=lattice_parameters, depth=12)

x = random.randint(random.PRNGKey(0), (input_size,), 0, 2)

params = model.init(random.PRNGKey(0), x)

# print(jax.tree_util.tree_map(lambda x: x.shape, params))

model_apply = jax.jit(lambda p, x: model.apply(p, x))

x = random.randint(random.PRNGKey(0), (input_size,), 0, 2)
print(model_apply(params, x))
x = random.randint(random.PRNGKey(1), (input_size,), 0, 2)
print(model_apply(params, x))
x = random.randint(random.PRNGKey(2), (input_size,), 0, 2)
print(model_apply(params, x))
x = random.randint(random.PRNGKey(3), (input_size,), 0, 2)
print(model_apply(params, x))
x = random.randint(random.PRNGKey(4), (input_size,), 0, 2)
print(model_apply(params, x))
x = random.randint(random.PRNGKey(5), (input_size,), 0, 2)
print(model_apply(params, x))
x = random.randint(random.PRNGKey(6), (input_size,), 0, 2)
print(model_apply(params, x))
x = random.randint(random.PRNGKey(7), (input_size,), 0, 2)
print(model_apply(params, x))
x = random.randint(random.PRNGKey(8), (input_size,), 0, 2)
print(model_apply(params, x))
x = random.randint(random.PRNGKey(9), (input_size,), 0, 2)
print(model_apply(params, x))
