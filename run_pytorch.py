import argparse
import numpy as np
import torch
import torchvision

from torch.optim import AdamW, SGD
import torchvision.transforms as transforms

import resnet
from proxyprox import ProxyProx
from runner import run_proxyprox
from utils import load_data, seed_everything

parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument('--gpu-id', type=int, help='ID of the GPU to be used', default=0)
parser.add_argument('--opt', type=str, help='The optimizer to use', default='proxyprox')
args = parser.parse_args()
gpu_id = args.gpu_id
opt_name = args.opt

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

batch_size = 128

trainloader, testloader, num_classes = load_data(batch_size=batch_size)
checkpoint = len(trainloader) // 3 + 1

n_seeds = 8
max_seed = 424242
np.random.seed(42)
seeds = [np.random.choice(max_seed, size=1, replace=False)[0] for _ in range(n_seeds)]

experiment = 'resnet18'
weight_decay = 0

if experiment == 'densenet121':
    net_class = densenet.DenseNet121
elif experiment == 'resnet18':
    net_class = resnet.ResNet18
elif experiment == 'resnet50':
    net_class = resnet.ResNet50
   
   
dataset = 'cifar10'
criterion = torch.nn.CrossEntropyLoss()    
experiment += f'subset_bs_{batch_size}'
wandb_project = f'{dataset}_{experiment}'

transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

num_classes = 10
n_subset = 128 * 20
idx = torch.arange(n_subset)
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
subset = torch.utils.data.Subset(trainset, idx)
subsetloader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0)


n_epoch = 30
seed = int(seeds[gpu_id])
seed_everything(seed)
net = net_class()
net.to(device)

if opt_name == 'proxyprox':
    lr_in = 0.01
    lr = 1
    weight_decay = 5e-4
    momentum_in = 0
    momentum_out = 0
    momentum_estim = 0
    reg = 4
    n_epoch_in = 2
    l2 = True
    opt = ProxyProx(net.parameters(), lr_in=lr_in, lr=lr, momentum_in=momentum_in,
                    momentum_out=momentum_out, momentum_estim=momentum_estim,
                    weight_decay=weight_decay, reg=reg, l2=l2)
    method = f'pp_lr_{lr_in}_{lr}_reg_{reg}'
    if momentum_in > 0 or momentum_out > 0 or momentum_estim > 0:
        method += f'_m_{momentum_in}_{momentum_out}_{momentum_estim}'
    if n_epoch_in > 1:
        method += f'_e_{n_epoch_in}'
elif opt_name == 'sgd':
    lr = 0.1
    weight_decay = 5e-4
    l2 = True
    opt = SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    method = f'sgd_lr_{lr}'
elif opt_name == 'adam':
    lr = 0.001
    weight_decay = 0.1
    l2 = False
    opt = AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    method = f'adam_lr_{lr}'
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epoch, eta_min=lr / 2000)

if scheduler is not None:
    method += f'_cos_{n_epoch}'
if weight_decay > 0:
    method += '_l2' if l2 else '_wd'
    method += f'_{weight_decay}'

if opt_name == 'proxyprox':
    run_proxyprox(
        net=net, trainloader=trainloader, testloader=testloader, subsetloader=subsetloader,
        device=device, n_epoch=n_epoch, optimizer=opt, noisy_train_stat=False, run_name=method,
        scheduler=scheduler, use_wandb=True, checkpoint=60, wandb_project=wandb_project, n_epoch_in=n_epoch_in
    )
else:
    run(
        net=net, trainloader=trainloader, testloader=testloader, 
        device=device, n_epoch=n_epoch, optimizer=opt, noisy_train_stat=False, run_name=method,
        scheduler=scheduler, use_wandb=True, checkpoint=60, wandb_project=wandb_project
    )
