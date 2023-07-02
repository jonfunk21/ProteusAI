import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
import pandas as pd
import sys
sys.path.insert(0, '../../src')
import proteusAI.ml_tools.torch_tools as torch_tools

script_path = os.path.dirname(os.path.realpath(__file__))
train_plots_path = os.path.join(script_path, 'plots/train')
checkpoints_path = os.path.join(script_path, 'checkpoints')

class RegDataset(Dataset):
    def __init__(self, data: pd.DataFrame, data_path: str):
        self.data = data
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        rep_name = self.data['mutant'].iloc[index]
        rep_path = os.path.join(self.data_path, rep_name + '.pt')
        
        x = torch.load(rep_path, map_location=self.device)
        y = torch.tensor(self.data['y'].iloc[index], dtype=torch.float32).to(self.device)
        return x, y

class VAEDataset(Dataset):
    def __init__(self, data, encoding_type='OHE'):
        self.data = data
        self.encoding_type = encoding_type

        assert encoding_type in ['OHE', 'BLOSUM50', 'BLOSUM62']

        if encoding_type == 'OHE':
            self.min_val, self.max_val = (0, 1)
            self.encoder = torch_tools.one_hot_encoder
        if encoding_type == 'BLOSUM50':
            self.min_val, self.max_val = (-5.0, 15.0)
            self.encoder = lambda x: torch_tools.blosum_encoding(x, matrix='BLOSUM50')
        if encoding_type == 'BLOSUM62':
            self.min_val, self.max_val = (-4.0, 11.0)
            self.encoder = lambda x: torch_tools.blosum_encoding(x, matrix='BLOSUM62')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data['mutated_sequence'].iloc[index]

        # Encode the sequence using the assigned encoder
        x = self.encoder(sequence)

        if self.min_val != 0 and self.max_val != 1:
            x = (x - self.min_val) / (self.max_val - self.min_val)

        return x
    
# TODO: create custom datasets for OHE and BLOSUM encoding
    
def criterion(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(train_data, val_data, model, optimizer, criterion, scheduler, epochs, 
              device, model_name, verbose=False, script_path=script_path, plots_path=train_plots_path,
              checkpoints_path=checkpoints_path, save_checkpoints=False):
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
            if save_checkpoints:
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

def train_regression(train_data, val_data, model, optimizer, criterion, scheduler, epochs, 
                     device, model_name, verbose=False, script_path=script_path, 
                     plots_path=train_plots_path, checkpoints_path=checkpoints_path,
                     save_checkpoints=False):
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    pbar = tqdm(range(epochs), desc='Training')
    for epoch in pbar:
        model.train()
        train_loss = 0
        num_examples = 0
        for batch, targets in train_data:
            # Move the batch tensors to the right device
            batch = batch.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch).squeeze(1)
            loss = criterion(outputs, targets)
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
            for batch, targets in val_data:
                batch = batch.to(device)
                targets = targets.to(device)
                outputs = model(batch).squeeze(1)
                loss = criterion(outputs, targets)
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
            if save_checkpoints:
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
    name = fname.split('/')[-1].split('.')[0].replace('.', ' ')
    plt.figure(figsize=(10, 5))
    sns.lineplot(range(len(train_losses)), train_losses, label='Train Loss', color='blue')
    sns.lineplot(range(len(val_losses)), val_losses, label='Validation Loss', color='orange')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Model')
    plt.title(f'Train and Validation Losses {name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()

def plot_predictions_vs_groundtruth(val_data, model, device, fname=None):
    model.eval() # Set the model to evaluation mode

    predictions = []
    groundtruth = []

    with torch.no_grad(): # No need to track gradients
        for batch, targets in val_data:
            batch = batch.to(device)
            targets = targets.to(device)
            
            # Obtain model predictions
            outputs = model(batch).squeeze(1)
            predictions.extend(outputs.tolist())
            groundtruth.extend(targets.tolist())

    plt.figure(figsize=(10, 5))
    sns.scatterplot(groundtruth, predictions, alpha=0.5, color='orange')
    
    # Extract model name from fname and use it in the plot title
    if fname is not None:
        name = fname.split('/')[-1].split('.')[0].replace('_', ' ')
        plt.title(f'Predicted vs. True Activity Levels for {name}')
    else:
        plt.title('Predicted vs. True Activity Levels')
        
    plt.xlabel('True Activity Levels')
    plt.ylabel('Predicted Activity Levels')
    plt.plot([min(groundtruth), max(groundtruth)], [min(groundtruth), max(groundtruth)], color='grey', linestyle='dotted', linewidth=2)  # diagonal line
    plt.grid(True)

    if fname is not None:
        plt.savefig(fname)
    else:
        plt.show()

    return predictions