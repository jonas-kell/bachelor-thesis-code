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

from split_net import CombineToComplexModule, CombineToComplexNet

net1 = nn.Dense(3)
net2 = nn.Dense(3)

combiner = CombineToComplexNet(net_1=net1, net_2=net2)

x = jnp.ones((10, 1))
print(x)

params = combiner.init(random.PRNGKey(0), x)

x = combiner.apply(params, x)
print(x)
