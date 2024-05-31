import argparse
import sys
from typing import Any
from statistics import mean, stdev

import yaml
from scipy.stats import wasserstein_distance
import torch

from calc_obs import get_obs


def calc_wasserstein(
        reference:dict[str, dict[str, torch.Tensor|None|bool]],
        obs_list:dict[str, dict[str, torch.Tensor|None|bool]],
        num_splits:int=1):
    results = []
    for name in reference:
        obs_ref = reference[name]['obs']
        weights_ref = reference[name]['weights']
        if weights_ref is not None:
            weights_ref = weights_ref[torch.isfinite(obs_ref)]
        obs_ref = obs_ref[torch.isfinite(obs_ref)]

        r = torch.randperm(len(obs_ref))
        obs_ref = obs_ref[r]
        if weights_ref is not None:
            weights_ref = weights_ref[r]
        std_value, mean_value = torch.std_mean(obs_ref)
        obs_ref = (obs_ref - mean_value)/std_value

        obs_l = obs_list[name]['obs']
        weights_l = obs_list[name]['weights']
        if weights_ref is not None:
            weights_l = weights_l[torch.isfinite(obs_l)]
        obs_l = obs_l[torch.isfinite(obs_l)]

        r = torch.randperm(len(obs_l))
        obs_l = obs_l[r]
        if weights_l is not None:
            weights_l = weights_l[r]
        obs_l = (obs_l - mean_value)/std_value

        min_len = min(len(obs_l), len(obs_ref))
        split_size = -(-min_len//num_splits)
        obs_l = torch.split(obs_l[:min_len], split_size)
        obs_ref = torch.split(obs_ref[:min_len], split_size)
        if weights_ref is not None:
            weights_l = torch.split(weights_l[:min_len], split_size)
            weights_ref = torch.split(weights_ref[:min_len], split_size)

        if weights_ref is not None:
            wdist = [wasserstein_distance(e.numpy(), e_ref.numpy(), w.numpy(), w_ref.numpy()) 
                     for e, w, e_ref, w_ref in zip(obs_l, weights_l, obs_ref, weights_ref)]
        else:
            wdist = [wasserstein_distance(e.numpy(), e_ref.numpy()) 
                     for e, e_ref in zip(obs_l, obs_ref)]

        results.append(wdist)

        print(f'{name}, {mean(wdist):e}, {stdev(wdist):e}')
        sys.stdout.flush()

    return results

@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Where to find the data')
    parser.add_argument('-g', '--gent4', default='data/gamma_test.h5',
        help='Where to find the GENT4 data')
    parser.add_argument('--config', default='conf/observables.yaml',
        help='Where to find the config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    obs = get_obs(args.file, config)
    reference = get_obs(args.gent4, config)

    calc_wasserstein(reference, obs, num_splits=10)

if __name__=='__main__':
    torch.set_default_dtype(torch.float64)
    main()
