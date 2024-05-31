from typing import Any, TypeVar
import math

import torch
from torch import nn
import torch.nn.functional as F
from nflows import transforms

from torch_util import *

class UNet(nn.Module):

    def __init__(self,
            in_features:int,
            out_features:int,
            hidden:int,
            identity_init:bool=True,
            downsamples:list[tuple[int,...]]=[],
            cyclic_padding:bool=False
        ) -> None:
        super().__init__()

        self.cyclic_padding = cyclic_padding

        self.encoder_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for i, kernel_size in enumerate(downsamples):
            self.encoder_blocks.append(self.get_block(
                in_features if i==0 else 2**i*hidden,
                2**i*hidden
            ))
            self.decoder_blocks.append(self.get_block(
                2**(i+1)*hidden,
                2**i*hidden,
                out_features if i==0 else None,
                activation=(i!=0)
            ))
            self.downs.append(nn.Conv2d(
                2**i*hidden,
                2**(i+1)*hidden,
                kernel_size,
                stride=kernel_size
            ))
            # self.downs.append(nn.AvgPool2d(kernel_size))
            self.ups.append(nn.ConvTranspose2d(
                2**(i+1)*hidden,
                2**i*hidden,
                kernel_size,
                stride=kernel_size
            ))
            # self.ups.append(nn.Upsample(scale_factor=tuple(kernel_size)))
        self.activation = nn.LeakyReLU()

        if identity_init:
            self.decoder_blocks[0][-1].weight.data.zero_()
            self.decoder_blocks[0][-1].bias.data.zero_()

    def forward(self, x: torch.Tensor, context: list[torch.Tensor]) -> torch.Tensor:
        x = torch.cat((x,context), dim=-3)
        skips = []
        for block, down in zip(self.encoder_blocks, self.downs):
            x = block(x)
            skips.append(x)
            x = down(x)
        x = self.activation(x)
        for block, up in zip(reversed(self.decoder_blocks), reversed(self.ups)):
            x = up(x)
            x = torch.cat((x,skips.pop()), dim=-3)
            x = block(x)
        return x

    def get_block(
            self,
            in_features:int,
            features:int,
            out_features:int|None=None,
            activation:bool=True
        ) -> nn.Module:
        if out_features is None:
            out_features = features
        if self.cyclic_padding:
            modules = [
                CircularPad(1),
                nn.Conv2d(in_features, features, 3, padding=(1,0)),
                nn.BatchNorm2d(features),
                nn.LeakyReLU(),
                CircularPad(1),
                nn.Conv2d(features, out_features, 3, padding=(1,0))
            ]
        else:
            modules = [
                nn.Conv2d(in_features, features, 3, padding=1),
                nn.BatchNorm2d(features),
                nn.LeakyReLU(),
                nn.Conv2d(features, out_features, 3, padding=1)
            ]
        if activation:
            modules.append(nn.BatchNorm2d(out_features))
            modules.append(nn.LeakyReLU())
        return nn.Sequential(*modules)


def compose_embedding_net(
        in_features:int,
        out_features:int,
        squeeze:int|tuple[int,int],
        cyclic_padding:bool=False
    ) -> nn.Sequential:
    if isinstance(squeeze,int):
        squeeze = (squeeze,squeeze)
    layers = []
    layers.append(nn.Conv2d(in_features, out_features, kernel_size=1, padding=0))
    layers.append(nn.LeakyReLU())
    if not cyclic_padding:
        layers.append(nn.Conv2d(out_features, out_features, kernel_size=3, padding=1))
    else:
        layers.append(CircularPad(1))
        layers.append(nn.Conv2d(out_features, out_features, kernel_size=3, padding=(1,0)))
    layers.append(nn.LeakyReLU())
    if squeeze[0]<=2 and squeeze[1]<=2:
        layers.append(nn.Conv2d(out_features, out_features, kernel_size=squeeze, stride=squeeze))
        layers.append(nn.LeakyReLU())
    else:
        layers.append(nn.AvgPool2d(squeeze))
    if not cyclic_padding:
        layers.append(nn.Conv2d(out_features, out_features, kernel_size=3, padding=1))
    else:
        layers.append(CircularPad(1))
        layers.append(nn.Conv2d(out_features, out_features, kernel_size=3, padding=(1,0)))

    return nn.Sequential(*layers)


class NoEmbeddingNet(nn.Module):

    def __init__(self,
            squeeze:int|tuple[int,int]
        ) -> None:
        super().__init__()
        if isinstance(squeeze, int):
            squeeze = (squeeze,squeeze)
        self.squeeze = squeeze

    def forward(self, context):
        return context[...,::self.squeeze[0],::self.squeeze[1]]


def get_coupling(coupling:str, identity_init:bool=True):
    class_dict = {
        'additive': transforms.AdditiveCouplingTransform,
        'affine': transforms.AffineCouplingTransform,
        'linear': transforms.PiecewiseLinearCouplingTransform,
        'quadratic': transforms.PiecewiseQuadraticCouplingTransform,
        'cubic': transforms.PiecewiseCubicCouplingTransform,
        'rational_quadratic': transforms.PiecewiseRationalQuadraticCouplingTransform
    }
    args_dict = {
        'additive': {},
        'affine': {
            'scale_activation': lambda x : torch.exp(6.*(torch.sigmoid(2./3.*x) - 0.5))
        },
        'linear': {
            'tails': 'linear',
            'tail_bound': 15.
        },
        'quadratic': {
            'tails': 'linear',
            'tail_bound': 15.
        },
        'cubic': {
            'tails': 'linear',
            'tail_bound': 15.
        },
        'rational_quadratic': {
            'tails': 'linear',
            'tail_bound': 15.,
            'enable_identity_init': identity_init
        }
    }
    coupling_class = class_dict[coupling]
    coupling_args = args_dict[coupling]
    return coupling_class, coupling_args

def compose_trafo(
        input_shape:int,
        context_canals:int,
        squeeze:int|tuple[int,int],
        num_blocks:int,
        coupling:str='affine',
        identity_init:bool=True,
        use_act_norm:bool=True,
        use_one_by_ones:bool=True,
        using_cache:bool=False,
        subnet_args:dict[str,any]={}
    ) -> tuple[
        transforms.Transform,
        list[transforms.OneByOneConvolution],
        tuple[int,...]]:
    coupling_class, coupling_args = get_coupling(coupling, identity_init)
    one_by_ones = []
    trafos = [transforms.SqueezeTransform(squeeze)]
    hidden_shape = trafos[-1].get_output_shape(*input_shape)

    def get_subnet(in_features, out_features):
        return UNet(
            in_features=in_features+context_canals,
            out_features=out_features,
            **subnet_args
        )
    for _ in range(num_blocks):
        if use_act_norm:
            trafos.append(transforms.ActNorm(hidden_shape[0]))
        trafos.append(coupling_class(
            mask=torch.arange(hidden_shape[0])<(hidden_shape[0])//2,
            transform_net_create_fn=get_subnet,
            **coupling_args
        ))
        if use_one_by_ones:
            trafos.append(transforms.OneByOneConvolution(
                num_channels=hidden_shape[0],
                using_cache=using_cache,
                identity_init=identity_init
            ))
            one_by_ones.append(trafos[-1])
        else:
            while True:
                permutation = torch.randperm(hidden_shape[0])
                # only accept permutations that effect the spiting
                if torch.max(permutation[:hidden_shape[0]//2])>=hidden_shape[0]//2:
                    break
            trafos.append(transforms.Permutation(
                permutation=permutation
            ))
    trafo = transforms.CompositeTransform(trafos)

    return trafo, one_by_ones, hidden_shape


TFlow = TypeVar("TFlow", bound="Flow")

class Flow(nn.Module):

    def __init__(self,
            trafo:transforms.Transform,
            embedding_net:nn.Module,
            data_dim:tuple[int,...],
            sample_input:tuple[torch.Tensor,...],
            hidden_shape:tuple[int,...]|None=None
        ) -> None:
        super().__init__()

        self.data_dim = data_dim
        self.transform = trafo
        self.embedding_net = embedding_net
        self.sample_input = sample_input
        if hidden_shape is None:
            hidden_shape = data_dim
        self.hidden_shape = hidden_shape

        self._log_z = 0.5 * math.prod(data_dim) * math.log(2 * math.pi)

    def log_prop_gauss(self, z:torch.Tensor) -> torch.Tensor:
        return - 0.5*torch.sum(z**2, dim=tuple(range(1,z.dim()))) - self._log_z

    def forward(self, input:torch.Tensor, cond:torch.Tensor) -> torch.Tensor:
        cond = self.embedding_net(cond)
        z, j_det = self.transform(input, cond)
        return self.log_prop_gauss(z) + j_det

    def sample(self, context:torch.Tensor) -> torch.Tensor:
        size = context.shape[0]
        noise = torch.randn(size, *self.hidden_shape, device=context.device)
        embedded_context = self.embedding_net(context)
        samples, _ = self.transform.inverse(noise, embedded_context)
        return samples

    @classmethod
    def load(cls, file:str) -> TFlow:
        checkpoint = torch.load(file, map_location='cpu')
        flow = cls(**checkpoint['hyperparameter'])
        flow.load_state_dict(checkpoint['weights'])
        return flow

    def save(self, file:str) -> None:
        checkpoint = {
            'hyperparameter': self.get_hyperparameter(),
            'weights': self.state_dict()
        }
        torch.save(checkpoint, file)

    def get_hyperparameter(self) -> dict[str,Any]:
        raise NotImplementedError()


class ConvFlow(Flow):

    def __init__(self, data_dim:tuple[int,...], idx:int, conf:dict[str,any]) -> None:
        conf = conf.copy()
        if conf.get('num_layers_cond', None) is None:
            conf['num_layers_cond'] = 2**30

        use_embedding_net = idx!=0
        num_layers_cond = conf.get('num_layers_cond', None)
        context_features = min(num_layers_cond, idx if idx is not None else 0)+2
        if isinstance(conf['squeeze'], list):
            squeeze = tuple(conf['squeeze'])
        else:
            squeeze = conf['squeeze']

        transform, one_by_ones, hidden_shape = compose_trafo(
            input_shape=data_dim,
            context_canals=conf.get('out_features_embed', 0) if use_embedding_net else 2,
            squeeze=squeeze,
            num_blocks=conf['num_blocks'],
            use_act_norm=conf.get('use_act_norm', True),
            use_one_by_ones=conf.get('use_one_by_ones', True),
            coupling=conf.get('coupling_block', 'affine'),
            subnet_args=conf.get('subnet_args', {})
        )

        if use_embedding_net:
            embedding_net = compose_embedding_net(
                in_features=context_features,
                out_features=conf['out_features_embed'],
                squeeze=squeeze,
                cyclic_padding=conf.get('subnet_args', {}).get('cyclic_padding', False)
            )
        else:
            embedding_net = NoEmbeddingNet(squeeze)

        sample_input = (
            torch.randn(1,*data_dim),
            torch.randn(1,context_features,*data_dim[1:])
        )

        super().__init__(transform, embedding_net, data_dim, sample_input, hidden_shape)
        self.idx = idx
        self.conf = conf
        self.one_by_ones = one_by_ones

    def cache_one_by_ones(self):
        for one_by_one in self.one_by_ones:
            one_by_one.use_cache()
            one_by_one._check_forward_cache()
            one_by_one._check_inverse_cache()

    def get_hyperparameter(self) -> dict[str,Any]:
        return {
            'data_dim': self.data_dim,
            'idx': self.idx,
            'conf': self.conf
        }


def build_maf_transformation(
        module_params:dict[str,Any],
        data_dim:tuple[int,...]):
    num_dims = math.prod(data_dim)
    num_blocks = module_params['num_blocks']
    permutation = module_params.get('permutation', None)

    params_flow = {
        'num_bins': module_params['num_bins'],
        'tails': 'linear',
        'tail_bound': -math.log(module_params['alpha']),
        'min_bin_width': module_params['min_bin_width'],
        'min_bin_height': module_params['min_bin_height'],
        'min_derivative': module_params['min_derivative'],
        'context_features': module_params['context_features_trafo'],
        'hidden_features': module_params['hidden_features'],
        'num_blocks': module_params['num_layers'],
        'random_mask': False,
        'activation': nn.ReLU(),
        'dropout_probability': module_params.get('dropout', 0.),
        'use_batch_norm': False,
        'features': num_dims,
        'use_residual_blocks': module_params.get(
            'use_residual_blocks', False)
    }

    transformation = []
    for i in range(num_blocks):
        if permutation == 'reverse':
            transformation.append(
                transforms.ReversePermutation(features=num_dims))
        elif permutation == 'random':
            transformation.append(
                transforms.RandomPermutation(features=num_dims))
        elif permutation == 'alternate':
            if i%2==0:
                transformation.append(
                    transforms.ReversePermutation(features=num_dims))
            else:
                transformation.append(
                    transforms.RandomPermutation(features=num_dims))
        elif permutation == None:
            pass
        else:
            raise NotImplementedError(
                'The only supported permutations are "reverse", "random", "alternate" and '
                f'None. (permutation={permutation})')

        transformation.append(
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                **params_flow))

    transformation = transforms.CompositeTransform(transformation)
    return transformation


class MAFlow(Flow):

    def __init__(self,
            data_dim:tuple[int,...],
            conf:dict) -> None:
        conf = conf.copy()
        transform = build_maf_transformation(conf, data_dim)
        embedding_net = nn.Identity()
        sample_input = (
            torch.randn(1,*data_dim),
            torch.randn(1,1)
        )

        super().__init__(transform, embedding_net, data_dim, sample_input)
        self.conf = conf

    def get_hyperparameter(self) -> dict[str,Any]:
        return {
            'data_dim': self.data_dim,
            'conf': self.conf
        }
