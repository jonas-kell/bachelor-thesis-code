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
        return jax.lax.complex(
            x_1, x_2
        )  # this is sufficient to  get an amplitude/phase network, because the network reults get pushed through an exp() in jVMC internally. to be exact, this is the same thing as jVMC does HERE: https://github.com/markusschmitt/vmc_jax/blob/1eb1aa691aa5e9c0ee112aa5c71e8c2136f17d81/jVMC/vqs.py#L31


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
