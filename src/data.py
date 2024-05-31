from typing import Optional, Any
import itertools

import numpy as np
import torch
import torch.types
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler
import torch.nn.functional as F

import data_util as util
from preprocessing import Transformation, Identity


torch_to_np_dtype = {
    torch.bool: np.bool_,
    torch.uint8: np.uint8,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128
}

def rotate(x:torch.Tensor, r:int):
    out = torch.zeros_like(x)
    out[...,:x.shape[-1]-r] = x[...,r:]
    out[...,x.shape[-1]-r:] = x[...,:r]
    return out

def get_noise_distribution(
            noise_mode:Optional[str],
            noise_mean:Optional[float]=None,
            noise_std:Optional[float]=None,
            device:torch.types.Device=None
        ) -> torch.distributions.Distribution|None:
        to_tensor = lambda x: torch.tensor(x, dtype=torch.get_default_dtype(), device=device)
        if noise_mode is None:
            noise_distribution = None
        elif noise_mode == 'uniform':
            noise_distribution = torch.distributions.Uniform(
                low= to_tensor(0.),
                high=to_tensor(2. * noise_mean))
        elif noise_mode == 'normal' or noise_mode == 'gaussian':
            noise_distribution = torch.distributions.Normal(
                loc=  to_tensor(noise_mean),
                scale=to_tensor(noise_std))
        elif noise_mode == 'beta':
            beta = torch.distributions.Beta(
                concentration0=to_tensor(2.),
                concentration1=to_tensor(2.))
            scale = torch.distributions.transforms.AffineTransform(
                loc=  to_tensor(0.),
                scale=to_tensor(2. * noise_mean)
            )
            noise_distribution = torch.distributions.TransformedDistribution(
                base_distribution=beta,
                transforms=scale
            )
        elif noise_mode=='log_normal':
            noise_distribution = torch.distributions.LogNormal(
                loc=  noise_mean*torch.log(to_tensor(10.)),
                scale=noise_std *torch.log(to_tensor(10.)))
        else:
            raise NotImplementedError(
                f'Noise mode {noise_mode} is not implemented!')
        return noise_distribution

class ShowerDataset(Dataset):

    def __init__(
            self,
            data_file:str,
            norm_mode:Optional[str]=None,
            noise_mode:Optional[str]=None,
            noise_mean:Optional[float]=None,
            noise_std:Optional[float]=None,
            extra_noise_mode:Optional[str]=None,
            extra_noise_mean:Optional[float]=None,
            extra_noise_std:Optional[float]=None,
            apply_cut_depos_energy:bool=False,
            energy_threshold:Optional[float]=None,
            samples_trafo:Transformation=Identity(),
            cond_trafo:Transformation=Identity(),
            cond_trafo2:Optional[Transformation]=None,
            start:int=0,
            stop:Optional[int]=None,
            layer:Optional[int]=None,
            num_layers_cond:int=2**30,
            skip_zeros:bool=False,
            padding:Optional[tuple[int]|list[int]]=None,
            device:torch.types.Device=None,
            random_shift:bool=False
        ) -> None:
        super().__init__()

        if skip_zeros and layer is None:
            raise NotImplementedError('skip_zeros is implemented only for single layers.')
        if layer is None:
            incid_energies, samples = util.load(
                data_file,
                start=start,
                stop=stop,
                dtype=torch_to_np_dtype[torch.get_default_dtype()]
            )
        else:
            incid_energies, samples = util.load(
                data_file,
                start=start,
                stop=stop,
                layer_start=max(0, layer-num_layers_cond),
                layer_stop=layer+1,
                dtype=torch_to_np_dtype[torch.get_default_dtype()]
            )
            layer = samples.shape[1]-1
        self.data_file = data_file
        self.norm_mode = norm_mode
        self.incid_energies = torch.from_numpy(incid_energies)
        self.samples = torch.from_numpy(samples)
        if skip_zeros:
            mask = torch.any(torch.any(self.samples[:,layer]>energy_threshold, 2), 1)
            self.samples = self.samples[mask]
            self.incid_energies = self.incid_energies[mask]
        if padding:
            self.samples = F.pad(self.samples, tuple(padding))
        self.noise_mode = noise_mode
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        self.apply_cut_depos_energy = apply_cut_depos_energy
        self.energy_threshold = energy_threshold
        self.layer = layer
        self.num_layers_cond = num_layers_cond
        self.samples_trafo = samples_trafo
        self.cond_trafo = cond_trafo
        if cond_trafo2 is None:
            self.cond_trafo2 = cond_trafo
        else:
            self.cond_trafo2 = cond_trafo2
        self.start = start
        self.stop = stop
        self.noise_distribution = get_noise_distribution(
            noise_mode, noise_mean, noise_std, device)
        self.extra_noise_distribution = get_noise_distribution(
            extra_noise_mode, extra_noise_mean, extra_noise_std, device)
        if device:
            self.incid_energies = self.incid_energies.to(device)
            self.samples = self.samples.to(device)
        self.random_shift = random_shift
        self.samples_trafo(self.samples)

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, index) -> dict[str, torch.tensor]:
        single_idx = isinstance(index, int)
        if single_idx:
            index = [index]
        inc_energy = self.incid_energies[index].detach().clone()
        sample = self.samples[index].detach().clone()
        if self.random_shift:
            r = torch.randint(0,sample.shape[-1],()).item()
            sample = rotate(sample, r)
        input_sample = sample.clone()

        if self.extra_noise_distribution:
            extra_noise = self.extra_noise_distribution.sample(sample.shape).reshape(sample.shape)
            extra_noise.clamp_(0., self.energy_threshold)
            sample = sample + extra_noise*(sample<1e-8)
        if self.noise_distribution:
            noise = self.noise_distribution.sample(sample.shape).reshape(sample.shape)
            noise.clamp_(0., self.energy_threshold)
            sample = sample + noise
            input_sample = input_sample + noise
        if self.apply_cut_depos_energy:
            input_sample[input_sample < self.energy_threshold] = 0
        depos_energy = input_sample.sum(dim=(-2, -1))

        if self.norm_mode==None:
            pass
        elif self.norm_mode=='layer':
            sample = sample/(depos_energy[..., None, None]+self.energy_threshold)
        else:
            raise NotImplementedError(f"Norm mode {self.norm_mode} is not implemented. Implemented modes: 'layer'")

        sample = self.samples_trafo(sample)

        if self.layer is not None:
            train_sample = sample[:,[self.layer]]

            prev_ecal_cells = sample[:,:self.layer]
            cond_layer_e = depos_energy[:,[self.layer]]

            repeat = tuple(itertools.repeat(1, inc_energy.dim()))+train_sample.shape[-2:]
            train_context = torch.cat((
                prev_ecal_cells,
                self.cond_trafo2(cond_layer_e)[...,None,None].repeat(repeat),
                self.cond_trafo(inc_energy)[...,None,None].repeat(repeat)
            ), dim=1)
        else:
            train_sample = sample[:,None,...]
            repeat1 = tuple(itertools.repeat(1, depos_energy.dim()+1))+train_sample.shape[-2:]
            repeat2 = tuple(itertools.repeat(1, inc_energy.dim()))+train_sample.shape[-3:]
            train_context = torch.cat((
                self.cond_trafo2(depos_energy)[:,None,:,None,None].repeat(repeat1),
                self.cond_trafo(inc_energy)[:,:,None,None,None].repeat(repeat2)
            ), dim=1)

        if single_idx:
            train_sample = train_sample[0]
            train_context = train_context[0]
        return {
            'samples': train_sample,
            'context': train_context
        }

    def __str__(self) -> str:
        return \
f'''ILDDataset(
    data_file={self.data_file},
    start={self.start},
    stop={self.stop},
    noise_distribution={self.noise_distribution},
    extra_noise_distribution={self.extra_noise_distribution},
    apply_cut_depos_energy={self.apply_cut_depos_energy},
    energy_threshold={self.energy_threshold},
    layer={self.layer},
    num_layers_cond={self.num_layers_cond},
    random_shift={self.random_shift}
)'''


class EnergyDataset(Dataset):
    def __init__(
            self,
            data_file:str,
            noise_mode:Optional[str]=None,
            noise_mean:Optional[float]=None,
            noise_std:Optional[float]=None,
            samples_trafo:Optional[Transformation]=None,
            cond_trafo:Optional[Transformation]=None,
            start:int=0,
            stop:Optional[int]=None,
            device:torch.types.Device='cpu'
        ) -> None:
        super().__init__()

        to_tensor = lambda x: torch.tensor(x, device=device, dtype=torch.get_default_dtype())

        incid_energies, layer_energies = util.load(data_file, start=start, stop=stop, layer_e=True)
        self.data_file = data_file
        self.incid_energies = to_tensor(incid_energies)
        self.layer_energies = to_tensor(layer_energies)
        self.samples_trafo = samples_trafo
        self.cond_trafo = cond_trafo
        self.start = start
        self.stop = stop
        self.noise_distribution = get_noise_distribution(noise_mode, noise_mean, noise_std, device)

    def __len__(self) -> int:
        return self.layer_energies.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.tensor]:
        samples = self.layer_energies[index].detach().clone()
        context = self.incid_energies[index].detach().clone()

        noise = self.noise_distribution.sample(samples.shape).reshape(samples.shape)
        samples = samples + noise
        samples = self.samples_trafo(samples)

        context = self.cond_trafo(context)

        return {
            'samples': samples,
            'context': context
        }

    def __str__(self) -> str:
        return \
f'''EnergiesDataset(
    data_file={self.data_file},
    start={self.start},
    stop={self.stop},
    noise_distribution={self.noise_distribution}
)'''


def get_loaders(
        conf_datasets:dict[str,Any],
        conf_loaders:dict[str,Any],
        layer:int,
        stop:int=-1
    ) -> tuple[DataLoader,DataLoader]:

    _, layers_shape = util.get_shapes(conf_datasets['data_file'])
    data_len = layers_shape[0]
    if stop>0:
        data_len = min(data_len, stop)
    split = data_len - data_len//10

    conf_datasets = conf_datasets.copy()
    cls = conf_datasets.get('class', 'ShowerDataset')
    cls = {
        'ShowerDataset': ShowerDataset,
        'EnergyDataset': EnergyDataset
    }[cls]
    if cls == ShowerDataset:
        conf_datasets['layer'] = layer
    if cls == EnergyDataset and 'cond_trafo2' in conf_datasets:
        del conf_datasets['cond_trafo2']
    if cls == EnergyDataset and 'num_layers_cond' in conf_datasets:
        del conf_datasets['num_layers_cond']
    if 'class' in conf_datasets:
        del conf_datasets['class']

    train_dataset = cls(
        **conf_datasets,
        start=0,
        stop=split)
    val_dataset = cls(
        **conf_datasets,
        start=split,
        stop=data_len)

    sampler = BatchSampler(
        RandomSampler(train_dataset),
        batch_size=conf_loaders['batch_size'],
        drop_last=False)
    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=sampler,
        batch_size=None,
        pin_memory=conf_loaders['pin_memory']
    )

    sampler = BatchSampler(
        SequentialSampler(val_dataset),
        batch_size=conf_loaders['batch_size'],
        drop_last=False)
    val_loader = DataLoader(
        dataset=val_dataset,
        pin_memory=conf_loaders['pin_memory'],
        sampler=sampler,
        batch_size=None
    )
    return train_loader, val_loader
