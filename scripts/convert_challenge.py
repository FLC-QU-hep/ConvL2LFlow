import argparse
import json

import numpy as np
import h5py


def to_challenge(args):
    assert len(args.in_files)==1
    with h5py.File(args.in_files[0], mode='r') as in_file:
        key = list(in_file.keys())[0]
        energies = in_file[key]['energy'][:]*1e3
        showers = in_file[key]['layers'][:]

    if not args.keep_order:
        showers = showers.swapaxes(-2,-1)

    if args.threshold:
        showers[showers<(args.threshold*1e3)] = 0.

    if args.num_files==1:
        with h5py.File(args.out_file, mode='w') as out_file:
            out_file.create_dataset('incident_energies',
                            data=energies.reshape(len(energies), -1),
                            compression='gzip')
            out_file.create_dataset('showers',
                            data=showers.reshape(len(showers), -1),
                            compression='gzip')
        return
    for i in range(args.num_files):
        start = i*len(energies)//args.num_files
        stop  = (i+1)*len(energies)//args.num_files
        out_file_name = args.out_file.format(i+1)
        print(out_file_name, start, stop)
        with h5py.File(out_file_name, mode='w') as out_file:
            out_file.create_dataset('incident_energies',
                            data=energies[start:stop].reshape(stop-start, -1),
                            compression='gzip')
            out_file.create_dataset('showers',
                            data=showers[start:stop].reshape(stop-start, -1),
                            compression='gzip')

def from_challenge(args):
    energies_list = []
    showers_list = []
    for in_file_name in args.in_files:
        print(in_file_name, end=': ')
        with h5py.File(in_file_name, mode='r') as in_file:
            energies = in_file['incident_energies'][:]/1e3
            showers = in_file['showers'][:]
            energies = energies.reshape(len(energies), 1)
            showers = showers.reshape(len(showers), *args.shape)
            energies_list.append(energies)
            showers_list.append(showers)
            print(showers.shape)

    energies = np.concatenate(energies_list)
    showers = np.concatenate(showers_list)
    del energies_list, showers_list

    if not args.keep_order:
        showers = showers.swapaxes(-2,-1)

    print(showers.shape)

    with h5py.File(args.out_file, mode='w') as out_file:
        key = f'{showers.shape[-2]}x{showers.shape[-1]}'
        out_file.create_dataset(key+'/energy', data=energies)
        out_file.create_dataset(key+'/layers', data=showers)

def main():
    parser = argparse.ArgumentParser(description='convert')
    parser.add_argument('in_files', nargs='+', help='input file path')
    parser.add_argument('out_file', help='output file path')
    parser.add_argument('-i', '--inverse', action='store_true')
    parser.add_argument('-s', '--shape', nargs=3, type=int, default=None)
    parser.add_argument('-n', '--num-files', type=int, default=1)
    parser.add_argument('--keep-order', action='store_true')
    parser.add_argument('-t', '--threshold', type=float, default=None)
    args = parser.parse_args()
    print(json.dumps(vars(args),indent=4))

    if args.inverse:
        to_challenge(args)
    else:
        from_challenge(args)


if __name__=='__main__':
    main()
