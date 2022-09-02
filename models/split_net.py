import jax
from jax.config import config

config.update("jax_enable_x64", True)
import flax.linen as nn
import jax.numpy as jnp


def complex_split_init(rng, shape):
    const = jax.nn.initializers.constant(1, dtype=jnp.complex64)

    result = const(rng, shape)
    result = result.at[0].set(1 + 0j)
    result = result.at[1].set(0 + 1j)

    return result


class CombineToComplexModule(nn.Module):
    """Marry the output of two real nets or two parts of one real net into one complex output"""

    def setup(self):
        self.factors = self.param(
            "factors",
            complex_split_init,
            (2,),
        )

    def __call__(self, x_1, x_2):
        return self.factors[0] * x_1 + self.factors[1] * x_2


class CombineToComplexNet(nn.Module):
    """Marry the output of two real nets into one complex output"""

    net_1: nn.Module
    net_2: nn.Module

    def setup(self):

        self.combiner = CombineToComplexModule()

    def __call__(self, x):
        x_1 = self.net_1(x)
        x_2 = self.net_2(x)

        return self.combiner(x_1, x_2)
