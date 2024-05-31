import sys
import argparse
import math

import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchmetrics
from torchinfo import summary
from sklearn.isotonic import IsotonicRegression

import data_util as util
from torch_util import *


class ConvolutionalClassifier(nn.Module):

    def __init__(self, use_bias, use_leaky_relu):
        super().__init__()

        if use_leaky_relu:
            Activation = nn.LeakyReLU
        else:
            Activation = nn.ReLU

        self.convs = nn.Sequential(
            nn.Unflatten(1, (1,-1)),
            nn.Conv3d(1, 16, 5, padding=3, bias=use_bias),
            Activation(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 16, 3, padding=1, bias=use_bias),
            Activation(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 16, 3, padding=1, bias=use_bias),
            Activation(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 16, 3, padding=1, bias=use_bias),
            Activation(),
            nn.MaxPool3d(2),
            nn.Flatten()
        )

        self.linears = nn.Sequential(
            nn.Linear(in_features=2*2*2*16+1, out_features=128, bias=use_bias),
            Activation(),
            nn.Linear(in_features=128, out_features=64, bias=use_bias),
            Activation(),
            nn.Linear(in_features=64, out_features=32, bias=use_bias),
            Activation(),
            nn.Linear(in_features=32, out_features=1, bias=use_bias),
            Activation()
        )

    def forward(self, x, cond):
        x = self.convs(x)
        x = torch.cat([x, cond], dim=1)
        x = self.linears(x)
        return x


class CircularConvolutionalClassifier(nn.Module):

    def __init__(self, use_bias, use_leaky_relu):
        super().__init__()

        if use_leaky_relu:
            Activation = nn.LeakyReLU
        else:
            Activation = nn.ReLU

        self.convs = nn.Sequential(
            nn.Unflatten(1, (1,-1)),
            CircularPad(2),
            nn.Conv3d(1, 16, 5, padding=(2,2,0), bias=use_bias),
            Activation(),
            CircularPad(1),
            nn.Conv3d(16, 16, 3, padding=(1,1,0), bias=use_bias),
            Activation(),
            nn.MaxPool3d((5,3,5)),
            CircularPad(2),
            nn.Conv3d(16, 16, 5, padding=(2,2,0), bias=use_bias),
            Activation(),
            nn.MaxPool3d((3,3,5)),
            CircularPad(1),
            nn.Conv3d(16, 16, 3, padding=(1,1,0), bias=use_bias),
            Activation(),
            nn.Flatten()
        )

        self.linears = nn.Sequential(
            nn.Linear(in_features=3*2*2*16+1, out_features=128, bias=use_bias),
            Activation(),
            nn.Linear(in_features=128, out_features=64, bias=use_bias),
            Activation(),
            nn.Linear(in_features=64, out_features=32, bias=use_bias),
            Activation(),
            nn.Linear(in_features=32, out_features=1, bias=use_bias),
            Activation()
        )

    def forward(self, x, cond):
        x = self.convs(x)
        x = torch.cat([x, cond], dim=1)
        x = self.linears(x)
        return x


class DenseClassifier(nn.Module):

    def __init__(self, in_features, hidden_features, use_bias, use_leaky_relu):
        super().__init__()

        if use_leaky_relu:
            Activation = nn.LeakyReLU
        else:
            Activation = nn.ReLU

        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features, use_bias),
            Activation(),
            nn.Linear(hidden_features, hidden_features, use_bias),
            Activation(),
            nn.Linear(hidden_features, hidden_features, use_bias),
            Activation(),
            nn.Linear(hidden_features, 1, use_bias)
        )

    def forward(self, x, cond:torch.Tensor):
        x = x.flatten(1)
        x = x/cond
        cond = torch.log10(cond)
        x = torch.cat([x, cond], dim=1)
        x = self.layers(x)
        return x


def get_datasets(file1, file2, cut=1e-4):
    len1 = util.get_shapes(file1)[0][0]
    len2 = util.get_shapes(file2)[0][0]
    num_samples = min(len1, len2)

    energies1, shower1 = util.load(file1, stop=num_samples)
    energies2, shower2 = util.load(file2, stop=num_samples)

    energies1 = torch.from_numpy(energies1).reshape(-1,1)
    shower1 = torch.from_numpy(shower1)
    energies2 = torch.from_numpy(energies2).reshape(-1,1)
    shower2 = torch.from_numpy(shower2)
    label1 = torch.zeros((num_samples, 1))
    label2 = torch.ones((num_samples, 1))

    energies = torch.cat((energies1, energies2))
    shower = torch.cat((shower1, shower2))
    label = torch.cat((label1, label2))

    shower[shower<cut] = 0.
    shower = 1e3*shower

    dataset = TensorDataset(shower, energies, label)
    train_data, val_data, test_data = random_split(dataset,[0.6,0.2,0.2])

    return train_data, val_data, test_data

def get_dataloaders(file1, file2, cut=1e-4, batch_size=256):
    train_dataset, val_dataset, test_data = get_datasets(file1, file2, cut)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=32
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2**11,
        shuffle=False,
        pin_memory=True,
        num_workers=32
    )
    test_loader = DataLoader(
        test_data,
        batch_size=2**11,
        shuffle=False,
        pin_memory=True,
        num_workers=32
    )
    return train_loader, val_loader, test_loader

def train_epoch(model, data, device, optimizer, criterion):
    model.train()
    average_loss = 0
    for showers, energies, label in data:
        showers = showers.to(device)
        energies = energies.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        scores = model(showers, energies)
        loss = criterion(scores, label)
        loss.backward()
        optimizer.step()

        average_loss += len(showers)*loss.item()
    average_loss /= len(data.dataset)
    return average_loss

@torch.no_grad()
def validation(model, data, device, criterion):
    model.eval()

    val_loss = 0
    all_label = []
    all_scores = []

    for showers, energies, label in data:
        showers = showers.to(device)
        energies = energies.to(device)
        label = label.to(device)

        scores = model(showers, energies)
        loss = criterion(scores, label)

        val_loss += len(showers)*loss.item()
        all_label.append(label)
        all_scores.append(torch.sigmoid(scores))

    all_label = torch.cat(all_label)
    all_scores = torch.cat(all_scores)
    val_loss /= len(data.dataset)

    return all_scores, all_label, val_loss

class EvaluateMetrics:

    def __init__(self, device):
        self.calc_auc = torchmetrics.AUROC(task='binary')
        self.calc_accuracy = torchmetrics.Accuracy(task='binary')
        self.calc_auc = self.calc_auc.to(device)
        self.calc_accuracy = self.calc_accuracy.to(device)

    @torch.no_grad()
    def __call__(self, all_scores, all_label, train_loss=None, val_loss=None, test_loss=None, name=None):
        auc = self.calc_auc(all_scores, all_label).item()
        acc = self.calc_accuracy(all_scores, all_label).item()
        jsd = calc_JSD(all_scores, all_label).item()
        jsd2 = calc_JSD(all_scores).item()

        if name:
            print(f'=== {name} ===')
        if train_loss:
            print('train loss:', train_loss)
        if val_loss:
            print('val loss:', val_loss)
        if test_loss:
            print('test loss:', test_loss)
        print('accuracy:', acc)
        print('AUC:', auc)
        print('JSD:', jsd)
        print('JSD w/o labels:', jsd2)
        # print('calibration_error:', calibration_error)
        print('')
        sys.stdout.flush()

def train(model, file1, file2, lr=1e-3, batch_size=256, epochs=30, device='cpu', cutoff=1e-4):
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr)
    train_loader, val_loader, test_loader = get_dataloaders(file1, file2, cutoff, batch_size)
    evaluate_metrics = EvaluateMetrics(device)

    showers, energies, _ = next(iter(train_loader))
    summary(model,  input_data=[showers.to(device), energies.to(device)])
    print('')

    to_numpy = lambda x: x.to(dtype=torch.float64, device='cpu').flatten().numpy()
    to_torch = lambda x: torch.from_numpy(x).to(dtype=torch.float32, device=device).unflatten(0,(-1,1))

    for i in range(epochs):
        train_loss = train_epoch(model, train_loader, device, optimizer, criterion)
        scores, labels, val_loss = validation(model, val_loader, device, criterion)
        evaluate_metrics(scores, labels, train_loss=train_loss, val_loss=val_loss, name=f'epoch {i:3d}')
        calibrator = IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1.-1e-6)
        calibrator.fit(to_numpy(scores), to_numpy(labels))
        scores_test, labels_test, test_loss = validation(model, test_loader, device, criterion)
        rescaled_scores = to_torch(calibrator.predict(to_numpy(scores_test)))
        evaluate_metrics(rescaled_scores, labels_test, test_loss=test_loss, name='test')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Where to find the data')
    parser.add_argument('-g', '--gent4', default='data/gamma_test.h5',
        help='Where to find the GENT4 data. Default: \'data/gamma_test.h5\'')
    parser.add_argument('-c', '--cutoff', default=1e-4, type=float,
        help='Cut of value for the data set in GeV. Default: 1e-4')
    parser.add_argument('-m', '--model', default='Conv',
        help="['Conv', 'CConv', 'Dense']. Default: 'Conv'")
    args = parser.parse_args()

    print('classifier type: low level')
    args_dict = vars(args)
    for arg in args_dict.keys():
        print(arg, args_dict[arg])
    print('')

    if args.model.strip().lower() == 'conv':
        model = ConvolutionalClassifier(use_bias=True, use_leaky_relu=True)
    elif args.model.strip().lower() == 'dense':
        energy_shape, sample_shape = util.get_shapes(args.file)
        model = DenseClassifier(
            in_features=math.prod(sample_shape[1:])+energy_shape[1],
            hidden_features=2048, use_bias=True, use_leaky_relu=True)
    elif args.model.strip().lower() == 'cconv':
        model = CircularConvolutionalClassifier(use_bias=True, use_leaky_relu=True)
    else:
        raise NotImplementedError(f"model {args.model} not implemented. Choose from ['Conv', 'CConv', 'Dense']")

    print(model)
    print('')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train(model, args.file, args.gent4,
        device=device,
        cutoff=args.cutoff,
        lr=0.001,
        batch_size=512)

if __name__=='__main__':
    main()
