import math

import torch
from torch import nn
import torch.nn.functional as F


__all__ = ['calc_JSD', 'CircularPad']

def calc_JSD(ratio, label=None):
    if label is None:
        label = ratio
    d1 = torch.mean(label*torch.log2(ratio+1e-8))
    d2 = torch.mean((1-label)*torch.log2(1.-ratio+1e-8))
    return d1+d2+1.


class CircularPad(nn.Module):

    def __init__(self, padding:int|tuple[int,int]=1) -> None:
        super().__init__()
        if isinstance(padding, int):
            self._padding = (padding,padding)
        else:
            self._padding = padding

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        shape = x.shape
        shape_temp = (math.prod(shape[:-2]),)+shape[-2:]
        shape_new = shape[:-1]+(shape[-1]+self._padding[0]+self._padding[1],)
        x = x.reshape(*shape_temp)
        x = F.pad(x, self._padding, 'circular')
        x = x.reshape(*shape_new)
        return x

    def extra_repr(self):
        return f'padding={self._padding}'
