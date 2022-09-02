from typing import Literal

from jax.config import config
import flax.linen as nn
import jVMC

config.update("jax_enable_x64", True)

from split_net import CombineToComplexNet


def cnn(
    inputs,
    ansatz: Literal[
        "single-real", "single-complex", "split-complex", "two-real"
    ] = "single-real",
) -> nn.Module:
    def myActFun(x):
        return 1 + nn.elu(x)

    channels = 16

    if ansatz == "single-real":
        net = jVMC.nets.CNN(
            F=(inputs,),
            channels=(channels,),
            strides=(1,),
            periodicBoundary=True,
            actFun=(myActFun,),
        )
    elif ansatz == "two-real":
        net = jVMC.nets.CNN(
            F=(inputs,),
            channels=(
                channels // 2,
            ),  # halfsize real net, as this later gets duplicated
            strides=(1,),
            periodicBoundary=True,
            actFun=(myActFun,),
        )
    elif ansatz == "single-complex":
        net = jVMC.nets.CpxCNN(
            F=(inputs,),
            channels=(channels,),
            strides=(1,),
            periodicBoundary=True,
            actFun=(myActFun,),
        )
    elif ansatz == "split-complex":
        net1 = jVMC.nets.CNN(
            F=(inputs,),
            channels=(channels // 2,),  # halfsize real net
            strides=(1,),
            periodicBoundary=True,
            actFun=(myActFun,),
        )
        net2 = jVMC.nets.CNN(
            F=(inputs,),
            channels=(channels // 2,),  # halfsize real net
            strides=(1,),
            periodicBoundary=True,
            actFun=(myActFun,),
        )
        net = CombineToComplexNet(net1, net2)
    else:
        raise RuntimeError(f"Ansatz '{ansatz}' ist not supported")

    return net


def rbm(
    ansatz: Literal[
        "single-real", "single-complex", "split-complex", "two-real"
    ] = "single-real",
) -> nn.Module:

    hidden = 8
    bias = False

    if ansatz == "single-real":
        net = jVMC.nets.RBM(numHidden=hidden, bias=bias)
    elif ansatz == "two-real":
        net = jVMC.nets.RBM(
            numHidden=hidden, bias=bias
        )  # halfsize real net, as this later gets duplicated
    elif ansatz == "single-complex":
        net = jVMC.nets.CpxRBM(numHidden=hidden, bias=bias)
    elif ansatz == "split-complex":
        net1 = jVMC.nets.RBM(numHidden=hidden // 2, bias=bias)  # halfsize real net
        net2 = jVMC.nets.RBM(numHidden=hidden // 2, bias=bias)  # halfsize real net
        net = CombineToComplexNet(net1, net2)
    else:
        raise RuntimeError(f"Ansatz '{ansatz}' ist not supported")

    return net
