import argparse
import numpy as np
import h5py

def main():
    parser = argparse.ArgumentParser(description='calculates and saves the layer energies')
    parser.add_argument('samples_file', help='samples file path')
    parser.add_argument('-t', '--threshold', type=float, default=1e-4, help='threshold')
    args = parser.parse_args()

    with h5py.File(args.samples_file, mode='r+') as h5_file:
        for key in h5_file.keys():
            print('process key:', key)
            if 'layer_e' in h5_file[key].keys():
                print('layer_e already exists.')
                overwrite = input('overwrite? [y/N]: ')
                if len(overwrite)>=1 and overwrite[0].lower()=='y':
                    print('overwrite.')
                    del h5_file[key]['layer_e']
                else:
                    print('skip.')
                    continue
            layers = h5_file[key]['layers'][:] / 1e3
            layers[layers<args.threshold] = 0.
            layers_e = np.sum(layers, axis=(2,3))
            dset = h5_file[key].create_dataset('layer_e', layers_e.shape, layers_e.dtype)
            dset[...] = layers_e

if __name__=='__main__':
    main()
