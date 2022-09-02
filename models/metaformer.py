from jax.config import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
import flax
import flax.linen as nn
import jVMC.nets.activation_functions as act_funs
from jax.lax import stop_gradient
from jVMC.nets.initializers import init_fn_args
import jVMC.global_defs as global_defs
import jax

from typing import Callable, Literal

from structures.lattice_parameter_resolver import LatticeParameters

from jax.experimental.host_callback import id_print

import jVMC
from functools import partial

complex_init = partial(jVMC.nets.initializers.cplx_init, dtype=jnp.complex64)

from split_net import CombineToComplexModule


class Identity(nn.Module):
    """Simple Identity Operator"""

    def __call__(self, x):
        return x


class AveragingConvolutionHead(nn.Module):
    @nn.compact
    def __call__(self, x):
        N, D = x.shape

        # average pooling to get from N,D -> 1,D -> D
        x = nn.avg_pool(x, window_shape=(N,))
        x = jnp.squeeze(x)

        return x


class Mlp(nn.Module):
    """Two layer MLP. Normally used for dims: I -> H -> I, where I=out_features and H=hidden_features"""

    hidden_features: int
    out_features: int
    act_layer: Callable = nn.gelu
    complex_values: bool = False

    def setup(self):
        self.fc1 = nn.Dense(
            self.hidden_features,
            **(  # complex init if wanted. Else default (real) init
                init_fn_args(
                    kernel_init=jVMC.nets.initializers.cplx_init,
                    bias_init=jax.nn.initializers.zeros,
                    dtype=global_defs.tCpx,
                )
                if self.complex_values
                else {}
            ),
        )
        self.fc2 = nn.Dense(
            self.out_features,
            **(  # complex init if wanted. Else default (real) init
                init_fn_args(
                    kernel_init=jVMC.nets.initializers.cplx_init,
                    bias_init=jax.nn.initializers.zeros,
                    dtype=global_defs.tCpx,
                )
                if self.complex_values
                else {}
            ),
        )

    def __call__(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.fc2(x)
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
    complex_values: bool = False

    def setup(self):
        head_dim = self.embed_dim // self.num_heads
        self.scale = head_dim**-0.5

        # in: self.embed_dim;  out: self.embed_dim * 3
        self.qkv = nn.Dense(
            self.embed_dim * 3,
            use_bias=self.qkv_bias,
            **(  # complex init if wanted. Else default (real) init
                init_fn_args(
                    kernel_init=jVMC.nets.initializers.cplx_init,
                    bias_init=jax.nn.initializers.zeros,
                    dtype=global_defs.tCpx,
                )
                if self.complex_values
                else {}
            ),
        )

        # in: self.embed_dim;  out: self.embed_dim
        self.proj = nn.Dense(
            self.embed_dim,
            use_bias=True,
            **(  # complex init if wanted. Else default (real) init
                init_fn_args(
                    kernel_init=jVMC.nets.initializers.cplx_init,
                    bias_init=jax.nn.initializers.zeros,
                    dtype=global_defs.tCpx,
                )
                if self.complex_values
                else {}
            ),
        )

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
    complex_values: bool = False  # TODO

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


class GraphMaskPooling(nn.Module):
    """Graph-Mask implementation that uses averaging neighbor interaction in order to facilitate pooling"""

    lattice_parameters: LatticeParameters
    graph_layer: Literal["symm_nn", "symm_nnn"] = "symm_nn"

    # interaction scaling here has been chosen to be uniform. (1, 1, 1). This could be explored further
    def setup(self):
        self.pooling_interaction_matrix = (
            1 * self.lattice_parameters["adjacency_matrices"]["avg_self_matrix"]
        )

        if self.graph_layer in ["symm_nn", "symm_nnn"]:
            self.pooling_interaction_matrix += (
                1 * self.lattice_parameters["adjacency_matrices"]["avg_nn_matrix"]
            )

        if self.graph_layer == "symm_nnn":
            self.pooling_interaction_matrix += (
                1 * self.lattice_parameters["adjacency_matrices"]["avg_nnn_matrix"]
            )

        self.pooling_interaction_matrix = stop_gradient(self.pooling_interaction_matrix)

    def __call__(self, x):

        x = jnp.matmul(self.pooling_interaction_matrix, x)

        return x


class GraphMaskConvolution(nn.Module):
    """Graph-Mask implementation that uses averaging neighbor interaction in order to facilitate pooling"""

    lattice_parameters: LatticeParameters
    embed_dim: int
    graph_layer: Literal["symm_nn", "symm_nnn"] = "symm_nn"
    complex_values: bool = False  # TODO

    def setup(self):
        self.factors = self.param(
            "factors", nn.initializers.normal(), (3, self.embed_dim)
        )

    def __call__(self, x):
        res = jnp.einsum(
            "d,nd->nd",
            self.factors[0],
            jnp.matmul(
                self.lattice_parameters["adjacency_matrices"]["add_self_matrix"], x
            ),
        )

        if self.graph_layer in ["symm_nn", "symm_nnn"]:
            res += jnp.einsum(
                "d,nd->nd",
                self.factors[1],
                jnp.matmul(
                    self.lattice_parameters["adjacency_matrices"]["add_nn_matrix"], x
                ),
            )

        if self.graph_layer == "symm_nnn":
            res += jnp.einsum(
                "d,nd->nd",
                self.factors[2],
                jnp.matmul(
                    self.lattice_parameters["adjacency_matrices"]["add_nnn_matrix"], x
                ),
            )

        return x


class Block(nn.Module):
    embed_dim: int
    token_mixer: nn.Module
    mlp_ratio: float = 4.0
    act_layer: Callable = nn.gelu
    norm_layer: Callable = nn.LayerNorm
    complex_values: bool = False

    def setup(self):
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)

        self.norm1 = self.norm_layer(
            dtype=jnp.complex64 if self.complex_values else jnp.float32,
            param_dtype=jnp.complex64 if self.complex_values else jnp.float32,
        )
        self.norm2 = self.norm_layer(
            dtype=jnp.complex64 if self.complex_values else jnp.float32,
            param_dtype=jnp.complex64 if self.complex_values else jnp.float32,
        )
        self.mlp = Mlp(
            hidden_features=mlp_hidden_dim,
            out_features=self.embed_dim,
            act_layer=self.act_layer,
            complex_values=self.complex_values,
        )

    def __call__(self, x):
        y = self.norm1(x)
        y = self.token_mixer(y)
        x = x + y

        z = self.norm2(x)
        z = self.mlp(z)
        x = x + z

        return x


class TokenMixer(nn.Module):
    lattice_parameters: LatticeParameters
    # structure info
    embed_dim: int
    complex_values: bool = False
    # token mixing
    token_mixer: Literal[
        "attention",
        "pooling",
        "convolution",
    ] = "attention"
    mixing_symmetry: Literal["arbitrary", "symm_nn", "symm_nnn"] = "arbitrary"
    # attention specific arguments
    num_heads: int = 3
    qkv_bias: bool = True

    def setup(self):
        # ! attention
        if self.token_mixer == "attention":
            self.token_mixer_module = Attention(
                lattice_parameters=self.lattice_parameters,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                qkv_bias=self.qkv_bias,
                mixing_symmetry=self.mixing_symmetry,
                complex_values=self.complex_values,
            )

        # ! pooling
        elif self.token_mixer == "pooling":
            if self.mixing_symmetry in ["symm_nn", "symm_nnn"]:
                self.token_mixer_module = GraphMaskPooling(
                    lattice_parameters=self.lattice_parameters,
                    graph_layer=self.mixing_symmetry,
                )
            else:
                raise RuntimeError(
                    f"Mixing symmetry modifier {self.mixing_symmetry} not supported for Pooling token mixer"
                )

        # ! convolution
        elif self.token_mixer == "convolution":
            if self.mixing_symmetry not in ["symm_nn", "symm_nnn"]:
                raise RuntimeError(
                    f"Mixing symmetry modifier {self.mixing_symmetry} not supported for convolution"
                )
            self.token_mixer_module = GraphMaskConvolution(
                lattice_parameters=self.lattice_parameters,
                graph_layer=self.mixing_symmetry,
                embed_dim=self.embed_dim,
                complex_values=self.complex_values,
            )

        else:
            raise RuntimeError(
                f"Token mixing operation {self.token_mixer} not implemented"
            )

    def __call__(self, x):
        return self.token_mixer_module(x)


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
    complex_values: bool = False

    def setup(self):
        self.dense_layer = nn.Dense(
            self.embed_dim,
            **(  # complex init if wanted. Else default (real) init
                init_fn_args(
                    kernel_init=jVMC.nets.initializers.cplx_init,
                    bias_init=jax.nn.initializers.zeros,
                    dtype=global_defs.tCpx,
                )
                if self.complex_values
                else {}
            ),
        )

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


class Metaformer(nn.Module):
    """Metafromer architecture, based on the experiments conducted for graph-image-processing

    Initialization arguments:
        * ``lattice_parameters``: Info on the shape of the input used for patch-embedding

    """

    lattice_parameters: LatticeParameters
    embed_dim: int = 60  # ! must be divisible by num_heads
    embed_mode: Literal[
        "duplicate_spread", "duplicate_nn", "duplicate_nnn"
    ] = "duplicate_spread"
    depth: int = 12
    mlp_ratio: float = 4.0
    norm_layer: Callable = nn.LayerNorm
    act_layer: Callable = nn.gelu
    # token mixing
    token_mixer: Literal[
        "attention",
        "pooling",
        "convolution",
    ] = "attention"
    mixing_symmetry: Literal["arbitrary", "symm_nn", "symm_nnn"] = "arbitrary"
    # attention specific arguments
    num_heads: int = 6
    qkv_bias: bool = True
    ansatz: Literal[
        "single-real", "single-complex", "split-complex", "two-real"
    ] = "single-real"

    def setup(self):
        # get dtype
        self.complex_values = self.ansatz == "single-complex"

        # Embed input into Metaformer-space
        self.site_embed = SiteEmbed(
            embed_dim=self.embed_dim,
            lattice_parameters=self.lattice_parameters,
            embed_mode=self.embed_mode,
            complex_values=self.complex_values,
        )

        # Blocks
        self.blocks = nn.Sequential(
            [
                Block(
                    embed_dim=self.embed_dim,
                    token_mixer=TokenMixer(
                        token_mixer=self.token_mixer,
                        lattice_parameters=self.lattice_parameters,
                        embed_dim=self.embed_dim,
                        mixing_symmetry=self.mixing_symmetry,
                        num_heads=self.num_heads,
                        qkv_bias=self.qkv_bias,
                        complex_values=self.complex_values,
                    ),
                    mlp_ratio=self.mlp_ratio,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer,
                    complex_values=self.complex_values,
                )
                for i in range(self.depth)
            ]
        )

        # Norm
        self.norm = self.norm_layer(
            dtype=jnp.complex64 if self.complex_values else jnp.float32,
            param_dtype=jnp.complex64 if self.complex_values else jnp.float32,
        )

        # Pooling-dimension-reducing-head
        self.head = AveragingConvolutionHead()

        # Assemble complex number from result
        if self.ansatz == "split-complex":
            self.combiner = CombineToComplexModule()

            if self.embed_dim % 2 != 0:
                raise RuntimeError("Split-Complex Mode for now requires even embed-dim")

    def __call__(self, x):
        # print(x)
        x = self.site_embed(x)

        x = self.blocks(x)

        x = self.norm(x)

        x = self.head(x)

        # different behaviour, depending on the ansatz, the net is used in
        if self.ansatz in [
            "single-real",
            "two-real",
        ]:
            # activation functions to convert to scalar energy output
            return jnp.sum(act_funs.log_cosh(x))

        # first half of result gets used for the real part, second part for the imaginary one
        elif self.ansatz == "split-complex":
            x1 = x[0 : self.embed_dim // 2]
            x2 = x[self.embed_dim // 2 :]

            x1 = jnp.sum(act_funs.log_cosh(x1))
            x2 = jnp.sum(act_funs.log_cosh(x2))

            x = self.combiner(x1, x2)

            return x
        elif self.ansatz == "single-complex":
            # activation function may not have definition holes because complex values
            return jnp.sum(act_funs.poly6(x))
        else:
            raise RuntimeError("Impossible ansatz")

    # as the outermost module contains lists/dicts as parameters (e.g. lattice_parameters), it is not hashable.
    # this could cause problems if more than one instance of an unhashable modlue is used. But here it should be fine for now
    def __hash__(self):
        return id(self)


def transformer(lattice_parameters: LatticeParameters, **kwargs):
    return Metaformer(
        lattice_parameters=lattice_parameters,
        embed_mode="duplicate_spread",
        mixing_symmetry="arbitrary",
        token_mixer="attention",
        **kwargs,
    )


def graph_transformer_nn(lattice_parameters: LatticeParameters, **kwargs):
    return Metaformer(
        lattice_parameters=lattice_parameters,
        embed_mode="duplicate_nn",
        mixing_symmetry="symm_nn",
        token_mixer="attention",
        **kwargs,
    )


def graph_transformer_nnn(lattice_parameters: LatticeParameters, **kwargs):
    return Metaformer(
        lattice_parameters=lattice_parameters,
        embed_mode="duplicate_nnn",
        mixing_symmetry="symm_nnn",
        token_mixer="attention",
        **kwargs,
    )


def graph_conformer_nn(lattice_parameters: LatticeParameters, **kwargs):
    return Metaformer(
        lattice_parameters=lattice_parameters,
        embed_mode="duplicate_nn",
        mixing_symmetry="symm_nn",
        token_mixer="convolution",
        **kwargs,
    )


def graph_conformer_nnn(lattice_parameters: LatticeParameters, **kwargs):
    return Metaformer(
        lattice_parameters=lattice_parameters,
        embed_mode="duplicate_nnn",
        mixing_symmetry="symm_nnn",
        token_mixer="convolution",
        **kwargs,
    )


def graph_poolformer_nn(lattice_parameters: LatticeParameters, **kwargs):
    return Metaformer(
        lattice_parameters=lattice_parameters,
        embed_mode="duplicate_nn",
        mixing_symmetry="symm_nn",
        token_mixer="pooling",
        **kwargs,
    )


def graph_poolformer_nnn(lattice_parameters: LatticeParameters, **kwargs):
    return Metaformer(
        lattice_parameters=lattice_parameters,
        embed_mode="duplicate_nnn",
        mixing_symmetry="symm_nnn",
        token_mixer="pooling",
        **kwargs,
    )
