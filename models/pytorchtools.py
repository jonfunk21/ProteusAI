import torch
from torch.utils.data import Dataset
import esm
import numpy as np
from scipy.stats import pearsonr
import os

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

def validate(model, dataloader, criterion):
    model.eval()  # set model to evaluation mode
    total_loss = 0
    total_rmse = 0
    n_samples = 0
    predicted_values = []
    target_values = []
    with torch.no_grad():
        for names, seqs, y_target in dataloader:
            x = embedd(names, seqs, device=device, rep_layer=33)
            y_target = torch.unsqueeze(y_target, dim=1).to(device)
            # Forward pass
            out = model(x.to(device))
            loss = criterion(out, y_target)
            total_loss += loss.item() * x.size(0)
            total_rmse += np.sqrt(((out - y_target) ** 2).mean().item()) * x.size(0)
            predicted_values.extend(out.squeeze().cpu().numpy())
            target_values.extend(y_target.squeeze().cpu().numpy())
            n_samples += x.size(0)

    avg_loss = total_loss / n_samples
    avg_rmse = total_rmse / n_samples
    r, _ = pearsonr(predicted_values, target_values)
    return avg_loss, avg_rmse, r
