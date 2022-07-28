from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax
import flax.linen as nn
import jVMC.nets.activation_functions as act_funs

from typing import Literal

from structures.lattice_parameter_resolver import LatticeParameters

from jax.experimental.host_callback import id_print


class Identity(nn.Module):
    """Simple Identity Operator"""

    def __call__(self, x):
        return x


class Attention(nn.Module):
    """Default attention module of a Transformer

    Initialization arguments:
        * ``embed_dim``: Embed dimension. Needs to be a whole number multiple of num_heads
        * ``num_heads``: Number of attention heads to use. Needs to be a whole number
        * ``qkv_bias``: If there should be a bias used in the qkv calculation matrix
        * ``mixing_symmetry``: To decide which graph mode to use for graph-attention
    """

    lattice_parameters: LatticeParameters
    embed_dim: int = 15
    num_heads: int = 3
    qkv_bias: bool = True
    mixing_symmetry: Literal["arbitrary", "symm_nn", "symm_nnn"] = "arbitrary"

    def setup(self):
        head_dim = self.embed_dim // self.num_heads
        self.scale = head_dim**-0.5

        # in: self.embed_dim;  out: self.embed_dim * 3
        self.qkv = nn.Dense(self.embed_dim * 3, use_bias=self.qkv_bias)

        # in: self.embed_dim;  out: self.embed_dim
        self.proj = nn.Dense(self.embed_dim, use_bias=True)

        self.graph_projection = (
            Identity()  # arbitrary lets this behave as a Transformer, not a Graph-Transformer
            if self.mixing_symmetry == "arbitrary"
            else GraphMaskAttention(
                lattice_parameters=self.lattice_parameters,
                graph_layer=self.mixing_symmetry,
            )
        )

    def __call__(self, x):
        N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(N, 3, self.num_heads, C // self.num_heads)
            .transpose(1, 2, 0, 3)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(0, 2, 1)) * self.scale

        attn = self.graph_projection(attn)  # modifies to graph attention

        attn = nn.softmax(attn, axis=2)

        x = (attn @ v).transpose(0, 2, 1).reshape(N, C)

        x = self.proj(x)

        return x


class GraphMaskAttention(nn.Module):
    """Graph-Mask implementation that is compatible with the Attention Module"""

    lattice_parameters: LatticeParameters
    graph_layer: Literal["symm_nn", "symm_nnn"] = "symm_nn"

    def setup(self):
        self.factors = self.param("factors", nn.initializers.normal(), (3,))

    def __call__(self, x):
        matrix = (
            self.factors[0]
            * self.lattice_parameters["adjacency_matrices"]["add_self_matrix"]
        )

        if self.graph_layer in ["symm_nn", "symm_nnn"]:
            matrix += (
                self.factors[1]
                * self.lattice_parameters["adjacency_matrices"]["add_nn_matrix"]
            )

        if self.graph_layer == "symm_nnn":
            matrix += (
                self.factors[2]
                * self.lattice_parameters["adjacency_matrices"]["add_nnn_matrix"]
            )

        x = self.lattice_parameters["adjacency_matrices"]["zero_to_neg_inf_function"](
            jnp.einsum("ij,hij->hij", matrix, x)
        )

        return x


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

    def setup(self):
        self.dense_layer = nn.Dense(self.embed_dim)

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
                self.lattice_parameters["nn_spread_matrix"],
                x,
            )
        elif self.embed_mode == "duplicate_nnn":
            # sites x 1 -> sites x (1 + #nn + #nnn)
            x = jnp.einsum(
                "ijk,k->ji",
                self.lattice_parameters["nnn_spread_matrix"],
                x,
            )
        else:
            raise RuntimeError(f"embed_mode '{self.embed_mode}' is not supported")

        # normally done with a Convolution in one step, but here: 1. spread info   2. encode through one Dense layer
        # sites x ?? -> sites x embed_dim
        x = self.dense_layer(x)

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

    def setup(self):
        self.site_embed = SiteEmbed(
            embed_dim=self.embed_dim,
            lattice_parameters=self.lattice_parameters,
            embed_mode=self.embed_mode,
        )

    def __call__(self, x):
        x = self.site_embed(x)

        x = jnp.sum(x)  # TODO temporary, avoid nans in later calculations

        # activation functions to convert to scalar energy output
        return jnp.sum(act_funs.log_cosh(x))

    # as the outermost module contains lists/dicts as parameters (e.g. lattice_parameters), it is not hashable.
    # this could cause problems if more than one instance of an unhashable modlue is used. But here it should be fine for now
    def __hash__(self):
        return id(self)
