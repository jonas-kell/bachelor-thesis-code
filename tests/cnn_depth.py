import jax
from jax.config import config

config.update("jax_enable_x64", True)

import os
import sys

import jax.random as random
import jax.numpy as jnp
import flax.linen as nn


sys.path.append(os.path.abspath("./../"))
sys.path.append(os.path.abspath("./../models"))

from preconfigured import cnn

inputs = 20

model = cnn(inputs=inputs)

x = jnp.ones((inputs,))
print(x)

params = model.init(random.PRNGKey(0), x)

x = model.apply(params, x)
print(x)
