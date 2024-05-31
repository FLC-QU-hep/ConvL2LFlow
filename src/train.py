import os
import sys
import argparse
import signal

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from matplotlib import pyplot as plt

from config import Config

class Trainer:

    def __init__(self,
            conf:Config,
            layer:int,
            device:str|torch.device='cpu',
            stop:int=-1
        ) -> None:
        self.conf = conf
        self.flow = conf.get_flow(layer)
        self.flow = self.flow.to(device)
        self.device = device
        self.train_loader, self.val_loader = conf.get_loaders(layer, stop)
        self.layer = layer
        print(self.train_loader.dataset)
        print(self.val_loader.dataset)
        sys.stdout.flush()

        self.weight_decay = conf.conf_trainer.get('weight_decay', 0)
        self.learning_rate = conf.conf_trainer['learning_rate']
        self.scheduler_name = conf.conf_trainer.get('scheduler', None)
        self.num_epochs = conf.conf_trainer['num_epochs']
        self.grad_clip = conf.conf_trainer.get('grad_clip', None)

        self.configure_optimizer()

        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epoch = 0
        self.killed = False
        self.min_val_loss = float('inf')

        if self.layer is not None:
            self.checkpoint_file = self.conf.get_path(f'checkpoints/layer{self.layer:02d}.pt')
            self.flow_file = self.conf.get_path(f'flows/flow{self.layer:02d}.pt')
            self.flow_file_final = self.conf.get_path(f'flows/flow{self.layer:02d}_final.pt')
            self.layer_str = f'{self.layer:02d}'
        else:
            self.checkpoint_file = self.conf.get_path(f'checkpoints/energy.pt')
            self.flow_file = self.conf.get_path(f'flows/flow.pt')
            self.flow_file_final = self.conf.get_path(f'flows/flow_final.pt')
            self.layer_str = ''

        if os.path.exists(self.checkpoint_file):
            self.load()

    def configure_optimizer(self) -> None:
        if self.weight_decay > 0.:
            self.optimizer = optim.AdamW(
                params=self.flow.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                weight_decay=self.weight_decay)
        else:
            self.optimizer = optim.Adam(
                params=self.flow.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999))

        if self.scheduler_name is None:
            self.scheduler = None
            self.scheduler_interval = 'never'
        elif self.scheduler_name.lower() == 'Step'.lower():
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer=self.optimizer,
                step_size=self.num_epochs//3,
                gamma=0.1,
                verbose=True)
            self.scheduler_interval = 'epoch'
        elif self.scheduler_name.lower() == 'Exponential'.lower():
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                optimizer=self.optimizer,
                gamma=3e-3**(1./self.num_epochs),
                verbose=True)
            self.scheduler_interval = 'epoch'
        elif self.scheduler_name.lower() == 'OneCycle'.lower():
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=self.optimizer,
                max_lr=self.learning_rate,
                total_steps=self.num_epochs*len(self.train_loader),
                verbose=False)
            self.scheduler_interval = 'step'
        else:
            raise NotImplementedError(f'Scheduler {self.scheduler} not implemented.')

    def fit(self) -> None:
        for epoch in range(self.epoch+1, self.num_epochs+1):
            train_losses = []
            self.flow.train()
            for batch in self.train_loader:
                samples = batch['samples'].to(self.device)
                context = batch['context'].to(self.device)
                nll = - self.flow(samples, context)
                loss = torch.mean(nll)
                self.optimizer.zero_grad()
                loss.backward()
                if self.grad_clip:
                    clip_grad_norm_(self.flow.parameters(), self.grad_clip)
                self.optimizer.step()
                if self.scheduler_interval == 'step':
                    self.scheduler.step()
                train_losses.append(torch.sum(nll).item())
            if self.scheduler_interval == 'epoch':
                self.scheduler.step()
            val_losses = []
            self.flow.eval()
            with torch.no_grad():
                for batch in self.val_loader:
                    samples = batch['samples'].to(self.device)
                    context = batch['context'].to(self.device)
                    nll = - self.flow(samples, context)
                    val_losses.append(torch.sum(nll).item())
            mean_train_loss = sum(train_losses)/len(self.train_loader.dataset)
            mean_val_loss = sum(val_losses)/len(self.val_loader.dataset)
            self.train_losses.append(mean_train_loss)
            self.val_losses.append(mean_val_loss)
            self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            self.epoch = epoch
            self.print_and_plot()
            self.save()

    def print_and_plot(self) -> None:
        print(f'=== epoch {self.epoch:4d} ===')
        print(f'{"train loss:":16s} {self.train_losses[-1]:3.2f}')
        print(f'{"validation loss:":16s} {self.val_losses[-1]:3.2f}')
        print(f'{"learning rate:":16s} {self.learning_rates[-1]:.2e}')
        sys.stdout.flush()

        start = 10 if self.epoch > 20 else 0
        plt.plot(list(range(start,self.epoch)), self.train_losses[start:], label='train')
        plt.plot(list(range(start,self.epoch)), self.val_losses[start:], label='validation')
        plt.legend()
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.savefig(self.conf.get_path(f'plots/losses{self.layer_str}.pdf'), bbox_inches='tight')
        plt.close()

        plt.plot(list(range(self.epoch)), self.learning_rates)
        plt.ylabel('lr')
        plt.xlabel('epoch')
        plt.savefig(self.conf.get_path(f'plots/lr{self.layer_str}.pdf'), bbox_inches='tight')
        plt.close()

    def _signal_handler(self, sig, frame):
        self.killed = True

    def save(self) -> None:
        # ignore interruptions while writing checkpoints
        original_sigint_handler = signal.getsignal(signal.SIGINT)
        original_sigtherm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # save losses
        with open(self.conf.get_path(f'data/losses{self.layer_str}.txt'), 'a') as f:
            f.write(f'{self.train_losses[-1]} {self.val_losses[-1]}\n')

        # save checkpoint
        if self.scheduler is None:
            scheduler = {}
        else:
            scheduler = self.scheduler.state_dict()
        checkpoint = {
            'flow': self.flow.state_dict(),
            'epoch': self.epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'min_val_loss': self.min_val_loss,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': scheduler
        }
        if os.path.exists(self.checkpoint_file):
            os.remove(self.checkpoint_file)
        torch.save(checkpoint, self.checkpoint_file)

        if self.val_losses[-1] < self.min_val_loss:
            self.min_val_loss = self.val_losses[-1]
            if os.path.exists(self.flow_file):
                os.remove(self.flow_file)
            self.flow.save(self.flow_file)

        if self.num_epochs==self.epoch:
            self.flow.save(self.flow_file_final)

        # reinstate interruption handlers
        signal.signal(signal.SIGINT, original_sigint_handler)
        signal.signal(signal.SIGTERM, original_sigtherm_handler)
        if self.killed:
            print('exit')
            sys.exit(0)

    def load(self) -> None:
        try:
            checkpoint = torch.load(self.checkpoint_file, map_location=self.device)
        except:
            print(f'Loading {os.path.basename(self.checkpoint_file)} failed delete it.')
            sys.stdout.flush()
            os.remove(self.checkpoint_file)
            return

        self.flow.load_state_dict(checkpoint['flow'])
        self.epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.learning_rates = checkpoint['learning_rates']
        self.min_val_loss = checkpoint['min_val_loss']
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])

        print(f'Loaded {self.checkpoint_file} at epoch {self.epoch}.')
        sys.stdout.flush()



def get_args():
    parser = argparse.ArgumentParser(description='train network')
    parser.add_argument('param_file', type=str,
        help='where to find the parameters')
    parser.add_argument('-d', '--device', type=str, default='None',
        help='whether cuda should be used')
    parser.add_argument('-l', '--layer', type=int,
        help='calorimeter layer')
    parser.add_argument('--fast_dev_run', action='store_true', default=False,
        help='whether or not to use fast development run')
    return parser.parse_args()

def main():
    args = get_args()
    conf = Config(args.param_file, args.fast_dev_run)
    if args.device == 'None':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        if conf.conf_datasets.get('device', 'cpu') == 'cuda':
            conf.conf_datasets['device'] = args.device
    device = torch.device(device)
    # torch.backends.cuda.matmul.allow_tf32 = True
    if device.type == 'cuda':
        print('device name:', torch.cuda.get_device_name(device))
    print('device:', str(device))
    print('layer:', args.layer)
    print('result_path:', conf.result_path)
    sys.stdout.flush()
    if args.fast_dev_run:
        conf.conf_trainer['num_epochs'] = 2
        conf.conf_loader['batch_size'] = 8
        stop = 80
    else:
        stop = -1
    trainer = Trainer(conf, args.layer, device, stop)
    trainer.fit()

if __name__=='__main__':
    main()
