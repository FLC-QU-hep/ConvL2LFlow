import time
start = time.perf_counter()

import os
import argparse
from typing import Any, Optional
import itertools
import platform
import re
import sys
import warnings

import yaml
import torch
from torch import nn

import data_util
from preprocessing import compose
import flow
from postprocessing import Postprocessor


class Generator(nn.Module):

    def __init__(self,
            causal_path:str,
            energy_path:str,
            veto:float|None=None
        ) -> None:
        super().__init__()
        self.veto = veto

        causal_params = os.path.join(causal_path, 'conf.yaml')
        causal_flows_dir = os.path.join(causal_path, 'flows')
        energy_params = os.path.join(energy_path, 'conf.yaml')
        energy_flow_file = os.path.join(energy_path, 'flows/flow.pt')
        self.result_dir = causal_path

        with open(causal_params) as f:
            causal_params = yaml.load(f, Loader=yaml.FullLoader)
        with open(energy_params) as f:
            energy_params = yaml.load(f, Loader=yaml.FullLoader)

        self.num_layers_cond = causal_params['flow'].get('num_layers_cond', 2**30)
        self.energy_threshold = causal_params['dataset'].get('energy_threshold', 1e-4)
        self.noise_width = 2.*causal_params['dataset'].get('noise_mean', 5e-7)

        state_dict_files = self._get_checkpoint_files(causal_flows_dir)
        causal_flows = [flow.ConvFlow.load(file) for file in state_dict_files]
        self.causal_flows = nn.ModuleList(causal_flows)
        self.energy_flow = flow.MAFlow.load(energy_flow_file)

        for causal_flow in self.causal_flows:
            causal_flow.cache_one_by_ones()

        self.width, self.height = self.causal_flows[0].data_dim[-2:]
        self.num_layer = self.energy_flow.data_dim[0]
        assert self.num_layer == len(self.causal_flows)

        self.layer_flow_trafo_sample = compose(energy_params['dataset'].get('samples_trafo', None))
        self.layer_flow_trafo_cond = compose(energy_params['dataset'].get('cond_trafo', None))

        self.data_file = causal_params['dataset'].get('data_file', '')
        self._init_trafos(causal_params['dataset'])

        self.postprocess = causal_params.get('postprocess', {})
        self.spline_file = os.path.join(causal_path, 'spline.npy')

    @staticmethod
    def _get_checkpoint_files(state_dict_dir: str) -> list[str]:
        files = os.listdir(state_dict_dir)
        state_dict_files = []
        for file in files:
            if re.fullmatch(r'flow[0-9][0-9]_final.pt', file):
                state_dict_files.append(file)
        state_dict_files = sorted(state_dict_files)
        for i, state_dict_file in enumerate(state_dict_files):
            if int(state_dict_file[4:6]) != i:
                raise RuntimeError(
                    f'State dict file {state_dict_file} and index {i} do not match.'\
                    +' Have you trained all flows yet?')
        state_dict_files = [os.path.join(state_dict_dir, file) for file in state_dict_files]
        return state_dict_files

    def _init_trafos(self, params:dict[str,Any]) -> None:
        self.samples_trafo = compose(params.get('samples_trafo', None))
        self.cond_trafo = compose(params.get('cond_trafo', None))
        self.cond_trafo2 = compose(params.get('cond_trafo2', None))
        trafo_file = os.path.join(self.result_dir, 'trafo.pt')
        if os.path.isfile(trafo_file):
            state = torch.load(trafo_file, map_location='cpu')
            samples = torch.ones(1, self.num_layer, self.width, self.height)
            self.samples_trafo(samples)
            self.samples_trafo.load_state_dict(state)
        else:
            data_len = data_util.get_shapes(self.data_file)[1][0]
            split = data_len - data_len//10
            samples = data_util.load(self.data_file, stop=split)[1]
            samples = torch.from_numpy(samples)
            self.samples_trafo(samples)
            state = self.samples_trafo.state_dict()
            torch.save(state, trafo_file)

    def _sample_energy_flow(self, context:torch.Tensor) -> torch.Tensor:
        cond_transformed = self.layer_flow_trafo_cond(context)
        mask = torch.ones(context.shape[0],
            dtype=torch.bool,
            device=context.device)
        layer_energies = torch.zeros(context.shape[0],self.num_layer,
            dtype=torch.get_default_dtype(),
            device=context.device)
        while torch.count_nonzero(mask):
            layer_energies[mask] = self.energy_flow.sample(cond_transformed[mask])
            layer_energies[mask] = self.layer_flow_trafo_sample.inverse(layer_energies[mask])
            mask_c = mask.clone()
            if self.veto:
                mask[mask_c] = torch.sum(layer_energies[mask_c], dim=1)/context[mask_c].flatten()>self.veto
            else:
                mask = torch.zeros(context.shape[0], dtype=torch.bool, device=cond_transformed.device)
        return layer_energies

    def _sample_causal_flows(self, context:torch.Tensor) -> torch.Tensor:
        samples = []
        for idx, c_flow in enumerate(self.causal_flows):
            num_layers_cond = min(self.num_layers_cond, idx)
            cond_flow = samples[-num_layers_cond:]
            cond_flow.append(self.cond_trafo2(context[:,[idx]]))
            cond_flow.append(self.cond_trafo(context[:,[-1]]))

            for i in range(len(cond_flow)):
                if cond_flow[i].dim() < 3:
                    repeat = (tuple(itertools.repeat(1, cond_flow[i].dim()))
                            +(self.width, self.height))
                    cond_flow[i] = cond_flow[i][...,None,None].repeat(repeat)
            cond_flow = torch.cat(cond_flow, dim=1)

            samples_layer = c_flow.sample(cond_flow)
            samples.append(samples_layer)
        samples = torch.cat(samples, dim=1)
        samples = self.samples_trafo.inverse(samples)
        return samples

    def forward(self,
            context:torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
        layer_energies = self._sample_energy_flow(context)
        context = torch.cat((layer_energies, context), dim=1)
        samples = self._sample_causal_flows(context)
        return samples, layer_energies

def print_time(text):
    now = time.perf_counter()
    print(f'[{int(now-start):6d}s]: {text}')
    sys.stdout.flush()

# based on https://gitlab.com/Imahn/l2lflows/-/blob/main/auxiliary_scripts/functions_classes.py#L3414
def renormalize(showers:torch.Tensor, energies:torch.Tensor, threshold:float=1e-4):
    '''
    Do postprocessing on the raw sampled showers, which are then returned.
    Apply the new postprocessing.
    1) Sort the energy depositions in the ECal cells for a specific
        shower and layer.
    2) Apply a reverse cumulative sum along the dimension of the ECal
        cell energy depositions.
    3) Calculate renormalized showers by multiplying sorted showers with
        the sampled energies and dividing by the reverse cumulative sum
        from step 2). However, these are not the final showers we will
        use! Per shower and per layer, these renormalized showers are
        monotonically increasing.
    4) For the renormalized showers, find the indices where the
        renormalized showers are bigger or equal than the energy
        threshold. Per shower and layer, the smallest minimum index is
        taken.
    5) From the renormalized showers, calculate a renormalization factor
        with help of the minimum indices from 4). Basically, this
        renormalization factor comes from the ratio of the deposited
        energies over the reverse cumulative sum. To choose which ratio
        is taken, the minimum indices are used. Apply the renormalization
        on the unsorted (raw) showers.
    6) Construct a binary mask. Apply the binary mask on the showers from
        step 5). This mask effectively applies the energy threshold.

    :param Tensor showers: Raw showers sampled from shower flow in shape
        ``[num_samples, num_layers, x_cells, y_cells]``
    :param Tensor energies: Sampled energies by energy flow in shape
        ``[num_samples, num_layers]``
    :param float threshold: Energy threshold in GeV for detector readout
    :return: Postprocessed showers in shape ``[num_samples,
        num_layers, x_cells, y_cells]`` with ``energy_threshold`` already
        applied
    :rtype: Tensor
    '''
    # Shape of sorted showers: [num_samples, num_layers, x_cells*y_cells].
    sorted_showers, sort_indices = torch.sort(
        input=showers.reshape(showers.shape[0], showers.shape[1], -1),
        dim=2, descending=False, stable=False)

    # Shape of reverse_cumsum: [num_samples, num_layers, x_cells*y_cells].
    reverse_cumsum = sorted_showers.flip(dims=(2,)).cumsum(dim=2).flip(dims=(2,))

    ratio_ei_cumsum = energies.unsqueeze(dim=2) /\
                    reverse_cumsum
    renorm_showers = sorted_showers * ratio_ei_cumsum

    # Shapes of indices, min_indices and renorm_factor:
    # [num_samples, num_layers, x_cells*y_cells],
    # [num_samples, num_layers, 1] and [num_samples, num_layers, 1].
    indices = torch.where(
        (renorm_showers >= threshold),
        torch.arange(
            start=0, end=sorted_showers.shape[2], step=1, dtype=torch.int64
        ).repeat(showers.shape[0], showers.shape[1], 1),
        sorted_showers.shape[2] * torch.ones(
            *sorted_showers.shape, dtype=torch.int64))
    min_indices, _ = indices.min(dim=2, keepdim=True)

    # The concatenation with zero-tensor is happening so that in case of
    # min_indices containing the index x_cells*y_cells, a value can be
    # gathered.
    renorm_factor = torch.cat((
        ratio_ei_cumsum, torch.zeros(*min_indices.shape)), dim=2
    ).gather(dim=2, index=min_indices)
    showers_renormed = showers.reshape(
        *sorted_showers.shape) * renorm_factor

    # Shape of mask: [num_samples, num_layers, x_cells*y_cells].
    mask = torch.where(
        torch.arange(
            start=0, end=sorted_showers.shape[2], step=1,
            dtype=torch.int64
        ).repeat(showers.shape[0], showers.shape[1], 1) >= min_indices,
        torch.ones(
            *sorted_showers.shape, dtype=torch.int64),
        torch.zeros(
            *sorted_showers.shape, dtype=torch.int64)
    ).to(torch.bool)
    # Invert the permutation by using argsort():
    # https://discuss.pytorch.org/t/how-to-quickly-inverse-a-permutation-by-using-pytorch/116205/3
    # Inverse permutation is necessary because mask will be applied on
    # unsorted showers.
    mask = mask.gather(
        dim=2, index=sort_indices.argsort(dim=2, descending=False))
    showers_renormed[~mask] = 0
    showers_renormed = showers_renormed.reshape(*showers.shape)

    # Check for closeness. For this, sampled energies need an energy
    # threshold as well, otherwise ``torch.allclose()`` always returns
    # False.
    cut_sampled_energies = energies.clone()
    cut_sampled_energies[cut_sampled_energies < threshold] = 0.
    if not torch.allclose(cut_sampled_energies, showers_renormed.sum(dim=(2, 3))):
        warnings.warn('energies != sum(showers))', RuntimeWarning, stacklevel=2)

    if not torch.all(torch.eq(showers_renormed[showers_renormed<threshold], 0.)):
        warnings.warn('', RuntimeWarning, stacklevel=2)

    return showers_renormed

def generate(
        generator:Generator,
        context:torch.Tensor,
        batch_size:Optional[int]=None,
        device:str|torch.device='cpu'
    ) -> torch.Tensor:
    if batch_size is None:
        context = context.split(split_size=context.shape[0], dim=0)
    else:
        context = context.split(split_size=batch_size, dim=0)

    generator = generator.to(device)
    generator.eval()
    samples = []
    for batch, context_split in enumerate(context):
        print_time(f'start batch {batch:3d}')
        context_split = context_split.to(device)
        samples_l, energies_l = generator(context_split)
        samples_l = samples_l.cpu()
        energies_l = energies_l.cpu()
        if generator.postprocess.get('renormalize', False):
            samples_l = renormalize(
                showers=samples_l,
                energies=energies_l,
                threshold=generator.energy_threshold)
        samples.append(samples_l)
    samples = torch.cat(samples)
    print_time('generation done')
    return samples

def get_postprocessor(
        generator:Generator,
        context:torch.Tensor,
        batch_size:Optional[int]=None,
        device:str|torch.device='cpu'
    ) -> Postprocessor|None:
    if 'flow1d' not in generator.postprocess:
        return None
    postprocessor = Postprocessor(
        num_nodes=generator.postprocess['flow1d'].get('num_nodes',128),
        threshold=generator.energy_threshold,
        noise_width=generator.noise_width
    )
    if os.path.isfile(generator.spline_file):
        print_time('load spline for postprocessing')
        postprocessor.load_spline(generator.spline_file)
    else:
        print_time('load reference data to fit spline with')
        reference = data_util.load(generator.data_file, stop=100000)[1]
        print_time('generate samples to fit spline with')
        generated = generate(generator, context[:100000], batch_size, device)
        print_time('fit spline for postprocessing')
        postprocessor.init_spline(reference, generated)
        print_time('save spline')
        postprocessor.save_spline(generator.spline_file)
    return postprocessor

def get_args():
    parser = argparse.ArgumentParser(description='Generates new samples using a trained L2LFlows model.')
    parser.add_argument('energy_path',
        default=None,
        help='directory that contains the energy distribution flow weights')
    parser.add_argument('causal_path',
        help='directory that contains the causal flows weights and where the generated samples should be saved')
    parser.add_argument('-n', '--num-samples',
        default=1,
        type=int,
        help='number of samples to generate. default: 1')
    parser.add_argument('-b', '--batch-size',
        default=2**10,
        type=int,
        help='default: 2^10')
    parser.add_argument('--energy',
        default=None,
        type=int,
        help='fixed energy value')
    parser.add_argument('--veto',
        default=None,
        type=float,
        help='veto energy ratio threshold. default: None')
    parser.add_argument('-l', '--log',
        action='store_true', default=False,
        help='if set energies will be drawn from a log uniform distribution between 10 GeV and 1 TeV')
    parser.add_argument('--energy-file',
        default=None,
        help='given energy values as torch file')
    parser.add_argument('-d', '--device',
        default=None,
        help='device for computations')
    return parser.parse_args()

@torch.inference_mode()
def main():
    args = get_args()
    print_time('start main')
    print(yaml.dump(vars(args)), end='')
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'cuda' in device.lower():
        print('devise:', torch.cuda.get_device_name(torch.device(device)))
    elif device.lower() == 'cpu':
        print('devise:', platform.processor())
    print('num threads:', torch.get_num_threads())

    generator = Generator(args.causal_path, args.energy_path, args.veto)
    if args.energy_file is not None:
        energies = torch.load(args.energy_file, map_location='cpu').to(torch.get_default_dtype())
    elif args.energy is not None:
        energies = args.energy*torch.ones((args.num_samples,1))
    elif args.log:
        energies = 10**(3.*torch.rand((args.num_samples,1)))
    else:
        energies = 10.+90.*torch.rand((args.num_samples,1))

    # generator = torch.jit.script(generator, example_inputs=[(energies[:1],)])
    postprocessor = get_postprocessor(generator, energies, args.batch_size, device)
    samples = generate(generator, energies, args.batch_size, device)

    energies = energies.numpy()
    samples = samples.numpy()

    if postprocessor:
        print_time('start postprocessing')
        samples = postprocessor(samples)
        print_time('postprocessing done')

    for i in range(100):
        if args.energy is not None:
            name = f'samples{i:02d}_{args.energy:d}GeV'
        else:
            name = f'samples{i:02d}'
        if not os.path.exists(os.path.join(args.causal_path, name+'.h5')):
            break

    data_util.save(
        energy=energies,
        layers=samples,
        file=os.path.join(args.causal_path, name+'.h5'))

    print_time('all done')

if __name__=='__main__':
    main()
