import jax
from jax.config import config

config.update("jax_enable_x64", True)
import flax.linen as nn
import jax.numpy as jnp


class CombineToComplexModule(nn.Module):
    """Marry the output of two real nets or two parts of one real net into one complex output"""

    def setup(self):
        pass
        # this had the real-complex converters also as learnable parameters.
        # This caused a multitude of problems in the backpropagation, therefore this is now hardcoded and strict.
        # The previous layers will need to pick up handling the total scaling and the ratio of real to complex

    def __call__(self, x_1, x_2):
        return (1 + 0j) * x_1 + (0 + 1j) * x_2


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
