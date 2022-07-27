from jax.config import config

config.update("jax_enable_x64", True)
from jax.lax import stop_gradient
import jax.numpy as jnp
import flax
import flax.linen as nn
import jVMC.nets.activation_functions as act_funs

from typing import Literal

from structures.lattice_parameter_resolver import LatticeParameters


class SiteEmbed(nn.Module):
    """Transforms the input of "sites x 0/1-spin-states" to a numeric array of size "sites x embed_dim"
    Uses the embed_mode parameter to determine the way the information is embedded

    Initialization arguments:
        * ``lattice_parameters``: Info on the shape of the input used for patch-embedding

    """

    embed_dim: int
    lattice_parameters: LatticeParameters
    embed_mode: Literal[
        "duplicate_spread", "duplicate_nn", "duplicate_nnn"
    ] = "duplicate_spread"

    @nn.compact
    def __call__(self, x):
        # Go from 0/1 representation to 1/-1 (needed, because 0 represents no interaction in 'duplicate_...')
        x = 2 * x - 1

        if self.embed_mode == "duplicate_spread":
            # sites -> sites x 1
            x = jnp.expand_dims(x, (1))
        elif self.embed_mode == "duplicate_nn":
            # sites x 1 -> sites x (1 + #nn)
            x = jnp.einsum(
                "ijk,k->ji",
                stop_gradient(self.lattice_parameters["nn_spread_matrix"]),
                x,
            )
        elif self.embed_mode == "duplicate_nnn":
            # sites x 1 -> sites x (1 + #nn + #nnn)
            x = jnp.einsum(
                "ijk,k->ji",
                stop_gradient(self.lattice_parameters["nnn_spread_matrix"]),
                x,
            )
        else:
            raise RuntimeError(f"embed_mode '{self.embed_mode}' is not supported")

        # normally done with a Convolution in one step, but here: 1. spread info   2. encode through one Dense layer
        # sites x ?? -> sites x embed_dim
        dense_layer = nn.Dense(self.embed_dim)
        x = dense_layer(x)

        return x

    # as the module contains lists/dicts (e.g. lattice_parameters), it is not hashable.
    # this could cause problems if more than one instance of an unhashable module is used. But here it should be fine for now
    def __hash__(self):
        return id(self)


class Metaformer(nn.Module):
    """Metafromer architecture, based on the experiments conducted for graph-image-processing

    Initialization arguments:
        * ``lattice_parameters``: Info on the shape of the input used for patch-embedding

    """

    lattice_parameters: LatticeParameters
    embed_dim: int = 5
    embed_mode: Literal[
        "duplicate_spread", "duplicate_nn", "duplicate_nnn"
    ] = "duplicate_spread"

    @nn.compact
    def __call__(self, x):
        x = SiteEmbed(
            embed_dim=self.embed_dim,
            lattice_parameters=self.lattice_parameters,
            embed_mode=self.embed_mode,
        )(x)

        x = nn.Dense(self.lattice_parameters["nr_sites"])(x)

        # activation functions to convert to scalar energy output
        return jnp.sum(act_funs.log_cosh(x))

    # as the outermost module contains lists/dicts as parameters (e.g. lattice_parameters), it is not hashable.
    # this could cause problems if more than one instance of an unhashable modlue is used. But here it should be fine for now
    def __hash__(self):
        return id(self)
