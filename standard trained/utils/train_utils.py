import torch
import numpy as np 
from tqdm import tqdm

from lightly.loss import NTXentLoss
from lightly.loss import BarlowTwinsLoss

import torchvision

def reproducibility(SEED):
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)


def reshape_output(model_output):
    batch_size = len(model_output)
    return model_output.reshape(batch_size,-1)


def run_training_loop(model, main_loader,criterion, opt, every_batch, epochs, device, scheduler=None):
    model = model.to(device)
    print("Starting Training")
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch in main_loader:
            x0, x1 = batch[0]
            x0 = x0.to(device)
            x1 = x1.to(device)
            if hasattr(model, 'extract_feat'):
                z0 = reshape_output(model.extract_feat(x0)[0])
                z1 = reshape_output(model.extract_feat(x1)[0])
            else:
                z0 = model(x0)
                z1 = model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            loss.backward()
            opt.step()
            opt.zero_grad()
            if scheduler is not None:
                if every_batch:             
                    scheduler.step()
        if scheduler is not None:
            if not(every_batch):             
                scheduler.step()
        avg_loss = total_loss / len(main_loader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

    return model

criterions = dict()
criterions['BT'] = BarlowTwinsLoss
criterions['SimCLR'] = NTXentLoss
criterions['Dino'] = NTXentLoss

def get_criterion(model_name):
    return criterions[model_name]