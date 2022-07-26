from jax.config import config
import flax.linen as nn
import jVMC

config.update("jax_enable_x64", True)


def cnn(inputs) -> nn.Module:
    def myActFun(x):
        return 1 + nn.elu(x)

    net = jVMC.nets.CNN(
        F=(inputs,),
        channels=(16,),
        strides=(1,),
        periodicBoundary=True,
        actFun=(myActFun,),
    )

    return net


def complexRBM() -> nn.Module:
    net = jVMC.nets.CpxRBM(numHidden=8, bias=False)

    return net
