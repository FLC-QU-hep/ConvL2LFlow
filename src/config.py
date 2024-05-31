import os
from typing import Any

import yaml
from torch.utils.data import DataLoader

from preprocessing import compose
import data_util as util
import flow
from data import get_loaders

class Config:

    def __init__(self, file:str, fast_dev_run:bool=False) -> None:

        with open(file) as f:
            conf_full = yaml.load(f, yaml.FullLoader)

        self.conf_datasets:dict[str,Any] = conf_full['dataset']
        self.conf_flow:dict[str,Any] = conf_full['flow']
        self.conf_loader:dict[str,Any] = conf_full['dataloader']
        self.conf_trainer:dict[str,Any] = conf_full['trainer']

        self.conf_datasets['num_layers_cond'] = self.conf_flow.get('num_layers_cond', 2**30)
        self.conf_datasets['samples_trafo'] = compose(self.conf_datasets.get('samples_trafo', None))
        self.conf_datasets['cond_trafo'] = compose(self.conf_datasets.get('cond_trafo', None))
        self.conf_datasets['cond_trafo2'] = compose(self.conf_datasets.get('cond_trafo2', None))

        if os.path.isfile(self.conf_datasets['data_file']):
            _, layers_shape = util.get_shapes(self.conf_datasets['data_file'])
            self.data_dim = layers_shape[-2:]
            self.num_layer = layers_shape[-3]
            self.data_len = layers_shape[0]
        else:
            self.data_dim = self.conf_datasets.get('data_dim', None)
            self.num_layer = self.conf_datasets.get('num_layer', None)
            self.data_len = None

        self.samples_trafo = self.conf_datasets['samples_trafo']
        self.cond_trafo = self.conf_datasets['cond_trafo']
        self.cond_trafo2 = self.conf_datasets['cond_trafo2']

        if 'result_path' in conf_full:
            self.result_path = conf_full['result_path']
        else:
            self.result_path = util.setup_result_path(conf_full['run_name'], file, fast_dev_run)

    def get_flow(self, layer:int|None=None) -> flow.TFlow:
        conf = self.conf_flow.copy()
        cls = conf.get('class', 'ConvFlow')
        if 'class' in conf:
            del conf['class']
        if cls == 'ConvFlow':
            return flow.ConvFlow(
                (1,)+self.data_dim,
                layer,
                conf
            )
        else:
            return flow.MAFlow(
                (self.num_layer,),
                conf
            )

    def get_loaders(self, layer:int|None=None, stop:int=-1) -> tuple[DataLoader,DataLoader]:
        return get_loaders(
            self.conf_datasets,
            self.conf_loader,
            layer,
            stop
        )

    def get_path(self, name:str) -> str:
        path = os.path.join(self.result_path, name)
        dir_path, _ = os.path.split(path)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        return path
