import jax
from jax.config import config
import flax
import flax.linen as nn

config.update("jax_enable_x64", True)


def metaformer_base() -> nn.Module:
    return None
