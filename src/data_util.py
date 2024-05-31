from typing import Optional
import os
import datetime
import shutil

import numpy as np
import numpy.typing as npt
import h5py

def get_key(keys:list[str]) -> str:
    if '30x30' in keys:
        return '30x30'
    elif '25x25' in keys:
        return '25x25'
    elif '10x10' in keys:
        return '10x10'
    else:
        return keys[0]

def get_shapes(
        file:str
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    with h5py.File(file, mode='r') as h5_file:
        key = get_key(list(h5_file.keys()))
        energy_shape = h5_file[key]['energy'].shape
        layers_shape = h5_file[key]['layers'].shape
    return energy_shape, layers_shape

def load(
        file:str,
        start:int=0,
        stop:Optional[int]=None,
        dtype:npt.DTypeLike=np.float32,
        layer_e:bool=False,
        layer_start:int=0,
        layer_stop:Optional[int]=None,
    ) -> tuple[npt.NDArray, npt.NDArray]:
    with h5py.File(file, mode='r') as h5_file:
        key = get_key(list(h5_file.keys()))
        energy = h5_file[key]['energy'][start:stop]
        if not layer_e:
            layers = h5_file[key]['layers'][start:stop, layer_start:layer_stop] / 1e3
        else:
            layers = h5_file[key]['layer_e'][start:stop, layer_start:layer_stop]
    return energy.astype(dtype), layers.astype(dtype)

def save(energy:npt.ArrayLike, layers:npt.ArrayLike, file:str) -> None:
    key = f'{layers.shape[-2]}x{layers.shape[-1]}'
    with h5py.File(file, mode='w') as h5_file:
        h5_file.create_dataset(
            name=key + '/energy',
            data=energy)
        h5_file.create_dataset(
            name=key + '/layers',
            data=1e3*layers)

def setup_result_path(run_name:str, conf_file:str, fast_dev_run:bool=False):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(script_dir)

    now = datetime.datetime.now()
    while True:
        full_run_name = now.strftime('%Y%m%d_%H%M%S') + '_' + run_name
        result_path = os.path.join(script_dir, 'results', full_run_name)
        if not os.path.exists(result_path):
            if not fast_dev_run:
                os.makedirs(result_path)
            else:
                result_path = os.path.join(script_dir, 'results/test')
                if os.path.exists(result_path):
                    shutil.rmtree(result_path)
                os.makedirs(result_path)
            break
        else:
            now += datetime.timedelta(seconds=1)

    with open(conf_file, 'r') as f:
        contents = f.readlines()

    contents.insert(1, f'result_path: {result_path}\n')

    with open(os.path.join(result_path, 'conf.yaml'), 'w') as f:
        contents = "".join(contents)
        f.write(contents)

    return result_path
