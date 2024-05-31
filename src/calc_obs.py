import inspect
import math
import os
from typing import Any, Callable, Optional

import numpy as np
import torch

import data_util as util


def calc_e_layer(data, layer=0, threshold=1e-4):
    layer = torch.clone(data['layers'][:,layer])
    layer[layer<threshold] = 0.
    return torch.sum(layer, dim=(1,2))

def calc_e_detector(data, threshold=1e-4):
    layers = torch.clone(data['layers'])
    layers[layers<threshold] = 0.
    return torch.sum(layers,(1,2,3))

def calc_e_parton(data):
    return data['energy'].flatten()

def calc_e_ratio(data, threshold=1e-4):
    e_detector = calc_e_detector(data, threshold)
    e_parton = calc_e_parton(data)
    return e_detector/e_parton

def calc_e_layer_normed(data, layer=0, threshold=1e-4):
    e_layer = calc_e_layer(data, layer, threshold)
    e_total = calc_e_detector(data, threshold)
    return e_layer/e_total

def calc_brightest_voxel(data, layer=0, N=1, threshold=1e-4):
    layer = torch.clone(data['layers'][:,layer])
    layer[layer<threshold] = 0.
    layer = layer.reshape((layer.shape[0], -1))
    layer = layer[layer.sum(dim=1)>1e-8]
    layer, _ = layer.sort(dim=1)
    return layer[:,-N]/layer.sum(dim=1)

def calc_occupancy(data, threshold=1e-4):
    data = data['layers']
    return (data >= threshold).to(torch.get_default_dtype()).mean((1, 2, 3))

def calc_spectrum(data, cut=1e-5):
    data = data['layers']
    return data[data >= cut]

def calc_occupancy_layer(data, layer=0, threshold=1e-4):
    layer = data['layers'][:,layer]
    return (layer >= threshold).to(torch.get_default_dtype()).mean((1, 2))

def calc_spectrum_layer(data, layer=0, cut=1e-5):
    layer = data['layers'][:,layer]
    return layer[layer >= cut]

def mean_std(data, axis=0, get_std=False, threshold=1e-4, layer=None):
    axis += 1
    if layer is None:
        data = torch.clone(data['layers'])
    else:
        data = torch.clone(data['layers'][:,[layer]])
    data[data<threshold] = 0
    pos = torch.arange(data.shape[axis])
    repeat_shape = list(data.shape)
    repeat_shape[axis] = 1
    repeat_shape[0] = 1
    shape = [1,1,1,1]
    shape[axis] = data.shape[axis]
    pos = torch.tile(pos.reshape(shape), repeat_shape)
    weight = data/torch.sum(data, (1,2,3), keepdims=True)
    mean = torch.sum(pos*weight, (1,2,3), keepdims=True)
    if not get_std:
        return mean.flatten()
    std = torch.sqrt(torch.sum((pos-mean)**2*weight, (1,2,3)))
    return std

def mean_std_circular(data, axis=0, get_std=False, threshold=1.515e-5, layer=None, radius=2.325):
    axis += 1
    if layer is None:
        data = torch.clone(data['layers'])
    else:
        data = torch.clone(data['layers'][:,[layer]])
    data[data<threshold] = 0
    Z = torch.arange(data.shape[1], dtype=torch.get_default_dtype())
    R = radius*(0.5+torch.arange(data.shape[2], dtype=torch.get_default_dtype()))
    A = (2*math.pi/data.shape[3])*torch.arange(data.shape[3], dtype=torch.get_default_dtype())
    ZZ,RR,AA = torch.meshgrid(Z,R,A, indexing='ij')
    if axis==1:
        pos = ZZ
    elif axis==2:
        pos = RR*torch.sin(AA)
    elif axis==3:
        pos = RR*torch.cos(AA)
    pos = pos.unflatten(0,(1,-1))
    e_dep = torch.sum(data, (1,2,3), keepdims=True)

    weight = data[e_dep.flatten()>0.]/e_dep[e_dep.flatten()>0.]
    mean = torch.sum(pos*weight, (1,2,3), keepdims=True)
    if not get_std:
        return mean.flatten()
    std = torch.sqrt(torch.sum((pos-mean)**2*weight, (1,2,3)))
    return std

def profile(data, axis=0, threshold=1e-4):
    axis += 1
    idx = torch.where(data['layers']>threshold)
    pos = idx[axis].to(torch.get_default_dtype())
    weight = data['layers'][idx]/len(data['layers'])
    return pos, weight

def get_data(file:str, stop:int|None) -> dict[str, torch.Tensor]:
    energy, layers = util.load(
        file=file,
        dtype=np.float64,
        stop=stop)
    energy = torch.from_numpy(energy)
    layers = torch.from_numpy(layers)
    return {'energy': energy, 'layers': layers}

def call(
        function:Callable,
        data:dict[str, torch.Tensor],
        kwargs:dict[str, Any],
        threshold:float
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    if 'threshold' in inspect.getfullargspec(function).args:
        kwargs['threshold'] = threshold
    ret = function(data, **kwargs)
    if isinstance(ret, tuple):
        return ret
    else:
        return (ret, None)

def to_version_tuple(version_str:str):
    return tuple(map(int, (version_str.split('.'))))

def get_obs(
        file:str,
        config:list[dict[str, Any]],
        no_cache:bool=False,
        threshold:float=1e-4,
        stop:int|None=None
    ) -> dict[str, dict[str, torch.Tensor|None|bool]]:
    cache = os.path.splitext(file)[0] + '.pt'
    if os.path.isfile(cache):
        if no_cache:
            os.remove(cache)
        else:
            result = torch.load(cache)
            version = to_version_tuple(result.get('version', '0.0'))
            min_version = to_version_tuple('0.1')
            if version<min_version:
                os.remove(cache)
            else:
                del result['version']
                return result

    data = get_data(file, stop)
    result = {'version': '0.1'}
    lookup_functions = globals()
    for element in config:
        function = lookup_functions[element['function']]
        kwargs = element.get('args', {})
        name = element.get('name', element['function'])
        name = name.replace(' ', '_')

        obs, weights = call(function, data, kwargs, threshold)

        result[name] = {
            'obs': obs,
            'weights': weights,
            'classifier': element.get('classifier', False)
        }

    torch.save(result, cache)
    del result['version']
    return result
