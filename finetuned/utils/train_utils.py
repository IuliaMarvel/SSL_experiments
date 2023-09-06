import torch
import numpy as np 
from tqdm import tqdm


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


def run_training_loop(model, main_loader,criterion, opt,  epochs, device, scheduler=None):
    print("Starting Training")
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch in main_loader:
            x0, x1 = batch[0]
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0 = reshape_output(model.extract_feat(x0)[0])
            z1 = reshape_output(model.extract_feat(x1)[0])
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            loss.backward()
            opt.step()
            opt.zero_grad()
            if scheduler is not None:             
                scheduler.step()
        avg_loss = total_loss / len(main_loader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
