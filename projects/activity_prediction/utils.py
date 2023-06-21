import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

script_path = os.path.dirname(os.path.realpath(__file__))
plots_path = os.path.join(script_path, 'plots/train')
checkpoints_path = os.path.join(script_path, 'checkpoints')

class VAEDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.data['label'].iloc[index]
        x = self.data['x'].iloc[index]
        return x
    
def criterion_BCE(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def criterion_MSE(recon_x, x, mu, logvar):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

def train_vae(train_data, val_data, model, optimizer, criterion, scheduler, epochs, 
              device, model_name, verbose=False, script_path=script_path, plots_path=plots_path,
              checkpoints_path=checkpoints_path):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    pbar = tqdm(range(epochs), desc='Training')
    for epoch in pbar:
        model.train()
        train_loss = 0
        num_examples = 0
        for batch in train_data:
            # Move the batch tensors to the right device
            batch = batch.to(device)
            batch = batch.view(batch.size(0), -1)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            
            loss = criterion(recon, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_examples += batch.size(0)
            
        average_train_loss = train_loss / num_examples
        train_losses.append(average_train_loss)
    
        if epoch % 100 == 0:
            if verbose:
                print(f"Epoch {epoch+1}, Train Loss: {average_train_loss:.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            num_examples = 0
            for batch in val_data:
                 # flatten the batch
                batch = batch.view(batch.size(0), -1)
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss = criterion(recon, batch, mu, logvar)
                val_loss += loss.item()
                num_examples += batch.size(0)

            average_val_loss = val_loss / num_examples
            val_losses.append(average_val_loss)
            if epoch % 100 == 0:
                if verbose:
                    print(f"Epoch {epoch+1}, Val Loss: {average_val_loss:.4f}")
                    
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'Train Loss': average_train_loss, 'Val Loss': average_val_loss, 'LR': current_lr})

        # Save model if it's the best so far
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            model_dest = os.path.join(checkpoints_path, model_name + '.pt')
            torch.save(model.state_dict(), model_dest)
            if verbose:
                print(f"Model saved at epoch {epoch+1}, Val Loss: {average_val_loss:.4f}")
            best_epoch = epoch
    
        scheduler.step()
    
    plot_dest = os.path.join(plots_path, model_name + '.png')
    os.makedirs(plots_path, exist_ok=True)
    plot_losses(train_losses, val_losses, best_epoch, fname=plot_dest)

    return model

def plot_losses(train_losses, val_losses, best_epoch, fname=None):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Model')
    plt.title('Train and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()
