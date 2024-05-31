from typing import Any
import math

import torch
from torch import nn


class Transformation(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def inverse(self, x:torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class Sequential(Transformation):

    def __init__(self, modules:list[Transformation]) -> None:
        super().__init__()
        self.sub_modules = nn.ModuleList(modules)

    def forward(self, x:torch.Tensor):
        for module in self.sub_modules:
            x = module.forward(x)
        return x

    def inverse(self, x:torch.Tensor):
        for module in self.sub_modules[::-1]:
            x = module.inverse(x)
        return x


class Identity(Transformation):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return x

    def inverse(self, x:torch.Tensor) -> torch.Tensor:
        return x


class Log(Transformation):

    def __init__(self, base:float=math.e, alpha:float=1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.log_base = math.log(base)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.log(x+self.alpha)/self.log_base

    def inverse(self, x:torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_base*x) - self.alpha


class LogIt(Transformation):

    def __init__(self, alpha:float=1e-6) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = (1-2*self.alpha)*x + self.alpha
        return torch.log(x/(1-x))

    def inverse(self, x:torch.Tensor) -> torch.Tensor:
        x = torch.sigmoid(x)
        return (x - self.alpha) / (1 - 2*self.alpha)


class Affine(Transformation):

    def __init__(self, a:float=1., b:float=0.) -> None:
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.a*x + self.b

    def inverse(self, x:torch.Tensor) -> torch.Tensor:
        return (x-self.b)/self.a


class Clamp(Transformation):

    def __init__(self, min:float=0., max:float=1.) -> None:
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return torch.clamp(x, self.min, self.max)

    def inverse(self, x:torch.Tensor) -> torch.Tensor:
        return x


class NormLayer(Transformation):

    addition: float

    def __init__(self, addition:float=0.) -> None:
        super().__init__()
        self.register_buffer('layer_max', None)
        self.addition = addition

    def init(self, data:torch.Tensor):
        dims = (0,) + tuple(range(2,data.dim()))
        layer_max = torch.amax(data, dim=dims, keepdim=True)
        self.layer_max = layer_max+self.addition

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layer_max is None:
            self.init(x)
        return x/self.layer_max

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x*self.layer_max


def compose(transformation:list[list[Any]]|None) -> Transformation:
    if transformation == None:
        return Identity()
    trafo_list = []
    attrs = globals()
    for element in transformation:
        Trafo = attrs[element[0]]
        assert issubclass(Trafo, Transformation)
        args = tuple(element[1:])
        trafo_list.append(Trafo(*args))
    return Sequential(trafo_list)
