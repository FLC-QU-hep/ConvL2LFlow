import sys
import argparse
from typing import Any

import yaml
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
import torchmetrics

import data_util as util
from torch_util import *
import calc_obs


def get_classifier(in_features:int):
    return nn.Sequential(
        nn.Linear(in_features, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 32),
        nn.LeakyReLU(),
        nn.Linear(32, 1)
    )

def get_obs(data:str, config:list[dict[str, Any]], cutoff:float, stop:int):
    obs_list = []
    obs = calc_obs.get_obs(
        file=data, 
        config=config,
        threshold=cutoff,
        stop=stop)
    for name, element in obs.items():
        if element['classifier']:
            obs_list.append(element['obs'].reshape(-1,1))
            print(name)
    return torch.cat(obs_list, dim=1)

def get_datasets(file1:str, file2:str, config_file:str, cutoff:float):
    len1 = util.get_shapes(file1)[0][0]
    len2 = util.get_shapes(file2)[0][0]
    num_samples = min(len1, len2)

    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    obs1 = get_obs(file1, config, cutoff, num_samples)
    obs2 = get_obs(file2, config, cutoff, num_samples)
    obs = torch.cat((obs1, obs2)).to(torch.get_default_dtype())

    label1 = torch.zeros(num_samples,1)
    label2 = torch.ones(num_samples,1)
    label = torch.cat((label1, label2))

    r = torch.randperm(len(obs))
    obs = obs[r]
    label = label[r]
    split = int(len(obs)*0.8)
    split = [split, len(obs)-split]
    obs_train, obs_val = torch.split(obs, split)
    label_train, label_val = torch.split(label, split)

    mean = torch.mean(obs_train, dim=0, keepdim=True)
    std = torch.std(obs_train, dim=0, unbiased=False, keepdim=True)
    obs_train -= mean
    obs_train /= std
    obs_val -= mean
    obs_val /= std

    train_data = TensorDataset(obs_train, label_train)
    val_data = TensorDataset(obs_val, label_val)

    return train_data, val_data

def get_dataloaders(file1:str, file2:str, config_file:str, cutoff:float):
    train_dataset, val_dataset = get_datasets(file1, file2, config_file, cutoff)
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        pin_memory=True,
        num_workers=16
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        pin_memory=True,
        num_workers=16
    )
    return train_loader, val_loader

def train(file1:str, file2:str, config_file:str, lr:float=1e-4, epochs:int=30, device:str='cpu', cutoff:float=1e-4):
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    train_loader, val_loader = get_dataloaders(file1, file2, config_file, cutoff)
    model = get_classifier(train_loader.dataset.tensors[0].shape[1])
    print(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    calc_auc = torchmetrics.AUROC(task='binary')
    calc_auc = calc_auc.to(device)
    calc_accuracy = torchmetrics.Accuracy(task='binary')
    calc_accuracy = calc_accuracy.to(device)

    for i in range(epochs):
        model.train()
        train_loss = 0
        for obs, label in train_loader:
            obs = obs.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            scores = model(obs)
            loss = criterion(scores, label)
            loss.backward()
            optimizer.step()

            train_loss += len(obs)*loss.item()
        train_loss /= len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            all_label = []
            all_scores = []

            for obs, label in val_loader:
                obs = obs.to(device)
                label = label.to(device)

                scores = model(obs)
                loss = criterion(scores, label)

                val_loss += len(obs)*loss.item()
                all_label.append(label)
                all_scores.append(torch.sigmoid(scores))

            all_label = torch.cat(all_label)
            all_scores = torch.cat(all_scores)
            val_loss /= len(val_loader.dataset)
            auc = calc_auc(all_scores, all_label).item()
            acc = calc_accuracy(all_scores, all_label).item()
            jsd = calc_JSD(all_scores, all_label).item()
            jsd2 = calc_JSD(all_scores).item()

        print(f'=== epoch {i:3d} ===')
        print('train loss:', train_loss)
        print('val loss:', val_loss)
        print('accuracy:', acc)
        print('AUC:', auc)
        print('JSD:', jsd)
        print('JSD w/o labels:', jsd2)
        print('')
        sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Where to find the data')
    parser.add_argument('-g', '--gent4', default='data/gamma_test.h5',
        help='Where to find the GENT4 data')
    parser.add_argument('--config', default='conf/observables.yaml',
        help='Where to find the config file')
    parser.add_argument('-c', '--cutoff', default=1e-4, type=float,
        help='Cut of value for the data set in GeV. Default: 1e-4')
    args = parser.parse_args()

    print('classifier type: high level')
    args_dict = vars(args)
    for arg in args_dict.keys():
        print(arg, args_dict[arg])
    print('')

    train(args.file, args.gent4, args.config, device='cpu', cutoff=args.cutoff)

if __name__=='__main__':
    main()
