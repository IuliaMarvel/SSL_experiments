def get_input_stats(DATASET):
    if DATASET == 'CIFAR10':
        data_mean, data_std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
    elif DATASET == 'CIFAR100':
        data_mean, data_std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)

    return data_mean, data_std

import torch
import torchvision
from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)

transform = BYOLTransform(
    view_1_transform=BYOLView1Transform(input_size=32, gaussian_blur=0.0),
    view_2_transform=BYOLView2Transform(input_size=32, gaussian_blur=0.0),
)
from lightly.data import LightlyDataset


def get_train_loader(DATASET):
    if DATASET == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10("datasets/cifar_10",transform=transform, download=True)
    if DATASET == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100("datasets/cifar_100",transform=transform, download=True)
    if DATASET == 'TinyImageNet':
        dataset = torchvision.datasets.ImageFolder('datasets/tiny-imagenet-200/train', transform=transform)
    
    dataset = LightlyDataset.from_torch_dataset(dataset, transform=transform)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader
    
from torchvision import transforms as T
from torch.utils.data import DataLoader

batch_size = 32

test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])



def get_train_val_loaders(DATASET,transform=test_transform):
    if DATASET == 'CIFAR10':
        train_data = torchvision.datasets.CIFAR10("datasets/cifar_10",train=True, transform=transform,download=True)
        val_data = torchvision.datasets.CIFAR10("datasets/cifar_10",train=False, transform=transform,download=True)
    if DATASET == 'CIFAR100':
        train_data = torchvision.datasets.CIFAR100("datasets/cifar_100",train=True, transform=transform,download=True)
        val_data = torchvision.datasets.CIFAR100("datasets/cifar_100",train=False, transform=transform,download=True)
    if DATASET == 'TinyImageNet':
        train_data = torchvision.datasets.ImageFolder('datasets/tiny-imagenet-200/train', transform=transform)
        val_data = torchvision.datasets.ImageFolder('datasets/tiny-imagenet-200/val', transform=transform)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=1, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, num_workers=1, drop_last=True)

    return train_loader, val_loader