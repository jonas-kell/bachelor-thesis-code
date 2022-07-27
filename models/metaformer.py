from jax.config import config
import jax.numpy as jnp
import flax
import flax.linen as nn
import jVMC.nets.activation_functions as act_funs


config.update("jax_enable_x64", True)

from structures.lattice_parameter_resolver import LatticeParameters


class Metaformer(nn.Module):
    """Metafromer architecture, based on the experiments conducted for graph-image-processing

    Initialization arguments:
        * ``lattice_parameters``: Info on the shape of the input used for patch-embedding

    """

    lattice_parameters: LatticeParameters

    @nn.compact
    def __call__(self, x):
        x = 2 * x - 1  # Go from 0/1 representation to 1/-1

        x = nn.Dense(self.lattice_parameters["nr_sites"])(x)

        # activation functions to convert to scalar energy output
        return jnp.sum(act_funs.log_cosh(x))

    # as the outermost module contains list (e.g. lattice_parameters), it is not hashable.
    # this could cause problems if more than one instance of an unhashable modlue is used. But here it should be fine for now
    def __hash__(self):
        return id(self)


def metaformer_base(**args) -> nn.Module:
    return Metaformer(**args)
