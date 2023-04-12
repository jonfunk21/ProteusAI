import torch
from torch.utils.data import Dataset
import esm
import numpy as np
from scipy.stats import pearsonr
import os
import matplotlib.pyplot as plt

# pytorch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load esm1v and attach to device
esm1v, alphabet = esm.pretrained.esm1v_t33_650M_UR90S()
esm1v.to(device)
esm1v.eval()  # disables dropout for deterministic results
batch_converter = alphabet.get_batch_converter()

def embedd(names, x, device, rep_layer=33):
    # create embeddings using esm1v
    batch_labels, batch_strs, batch_tokens = batch_converter(list(zip(names, x)))
    with torch.no_grad():
        results = esm1v(batch_tokens.to(device), repr_layers=[rep_layer], return_contacts=True)
    token_representations = results["representations"][rep_layer]

    return token_representations

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        names = self.data['Description'].iloc[index]
        x = self.data['Sequence'].iloc[index]
        y = torch.tensor(self.data['Data_normalized'].iloc[index], dtype=torch.float32).to(self.device)
        return names, x, y


def invoke(early_stopping, loss, model, implement=False):
    if implement == False:
        return False
    else:
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            return True


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def validate(model, dataloader, criterion):
    total_loss = 0
    total_rmse = 0
    n_samples = 0
    predicted_values = []
    target_values = []
    with torch.no_grad():
        for names, seqs, y in dataloader:
            x = embedd(names, seqs, device=device, rep_layer=33)
            y = torch.unsqueeze(y, dim=1).to(device)
            # Forward pass
            out = model(x.to(device))
            loss = criterion(out, y)
            total_loss += loss.item()
            total_rmse += np.sqrt(((out - y) ** 2).mean().item()) * x.size(0)
            predicted_values.extend(out.squeeze().cpu().numpy())
            target_values.extend(y.squeeze().cpu().numpy())
            n_samples += x.size(0)

    avg_loss = total_loss / n_samples
    avg_rmse = total_rmse / n_samples
    r, _ = pearsonr(predicted_values, target_values)
    return avg_loss, avg_rmse, r

# training loop
def train(model, train_loader, val_loader, loss_fn, optimizer, device, epochs, patience, save_path, train_log='train_log'):
    model.train()
    with open(os.path.join(save_path, train_log), 'w') as f:
        print('Begin training:', file=f)
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    early_stopping = EarlyStopping(patience=patience)
    n_samples = 0
    for epoch in range(epochs):
        running_loss = 0.0
        for names, seqs, y in train_loader:
            x = embedd(names, seqs, device=device, rep_layer=33) # (batch_size, seq_len, embedd_dim)
            y = torch.unsqueeze(y, dim=1) # (batch_size, 1)

            # train activity predictor
            optimizer.zero_grad()
            out = model(x.to(device)) # (batch_size, 1)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_samples += x.size(0)


        epoch_loss = running_loss / n_samples
        train_losses.append(epoch_loss)

        val_loss, val_rmse, val_pearson = validate(model, val_loader, loss_fn)
        print(val_loss)
        with open(os.path.join(save_path, train_log), 'a') as f:
            print(f"Epoch {epoch + 1}:: avg train loss: {epoch_loss:.4f}, avg val loss: {val_loss:.4f}, avg val RMSE: {val_rmse:.4f}, avg val pearson: {val_pearson:.4f}", file=f)

        val_losses.append(val_loss)
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(save_path, 'activity_model'))
            with open(os.path.join(save_path, train_log), 'a') as f:
                print('Saved best model', file=f)
            best_val_loss = val_loss

        if invoke(early_stopping, val_losses[-1], model, implement=True):
            model.load_state_dict(torch.load(os.path.join(save_path, 'activity_model'), map_location=device))
            break

    return model

# performance evaluation
def plot_losses(fname, train_loss, test_loss, burn_in=20):
    plt.figure(figsize=(15, 4))
    plt.plot(list(range(burn_in, len(train_loss))), train_loss[burn_in:], label='Training loss')
    plt.plot(list(range(burn_in, len(test_loss))), test_loss[burn_in:], label='Test loss')

    # find position of lowest testation loss
    minposs = test_loss.index(min(test_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Minimum Test Loss')

    plt.legend(frameon=False)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(fname)
    plt.close()