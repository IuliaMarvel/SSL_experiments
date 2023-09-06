from lightly.transforms.byol_transform import (
    BYOLTransform,
    BYOLView1Transform,
    BYOLView2Transform,
)
from lightly.transforms.simclr_transform import SimCLRTransform
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torchvision

transforms = dict()
transforms['BT'] = BYOLTransform(
    view_1_transform=BYOLView1Transform(input_size=32, gaussian_blur=0.0),
    view_2_transform=BYOLView2Transform(input_size=32, gaussian_blur=0.0),
)
transforms['SimCLR'] = SimCLRTransform(input_size=32, gaussian_blur=0.0)
transforms['Dino'] = SimCLRTransform(input_size=32, gaussian_blur=0.0)


def get_transforms(model_name):
    return transforms[model_name]


def get_main_loader(DATASET, model_name):
    transform = get_transforms(model_name)
    if DATASET == 'CIFAR10':
        dataset = torchvision.datasets.CIFAR10("datasets/cifar10",transform=transform, download=True)
    if DATASET == 'CIFAR100':
        dataset = torchvision.datasets.CIFAR100("datasets/cifar100",transform=transform, download=True)
    if DATASET == 'TinyImageNet':
        dataset = torchvision.datasets.ImageFolder('datasets/tiny-imagenet-200/train', transform=transform)
    
    # dataset = LightlyDataset.from_torch_dataset(dataset, transform=transform)

    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=1,
        pin_memory=True
    )
    
    return dataloader

test_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
batch_size = 32


def get_train_val_loaders(DATASET,transform=test_transform):
    if DATASET == 'CIFAR10':
        train_data = torchvision.datasets.CIFAR10("datasets/cifar10",train=True, transform=transform,download=True)
        val_data = torchvision.datasets.CIFAR10("datasets/cifar10",train=False, transform=transform,download=True)
    if DATASET == 'CIFAR100':
        train_data = torchvision.datasets.CIFAR100("datasets/cifar100",train=True, transform=transform,download=True)
        val_data = torchvision.datasets.CIFAR100("datasets/cifar100",train=False, transform=transform,download=True)
    if DATASET == 'TinyImageNet':
        train_data = torchvision.datasets.ImageFolder('datasets/tiny-imagenet-200/train', transform=transform)
        val_data = torchvision.datasets.ImageFolder('datasets/tiny-imagenet-200/val', transform=transform)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=1, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, num_workers=1, drop_last=True)

    return train_loader, val_loader
