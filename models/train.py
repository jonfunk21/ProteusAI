import sys
sys.path.append('../')
import pandas as pd
import torch
from activity_predictor import FFNN
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.pytorchtools import CustomDataset, embedd, validate
import os

# Hyper parameters (later all argparse)
epochs = 10
batch_size = 26
rep_layer = 33
input_size = 147 * 1280
hidden_layers = [1280]
output_size = 1
save_path = 'checkpoints'
data_path = '../example_data/directed_evolution/GB1/GB1.csv'

if not os.path.exists(save_path):
    os.mkdir(save_path)

# load data and split into train and val. first naive split, normalize activity
df = pd.read_csv(data_path)
df['Data_normalized'] = (df['Data'] - df['Data'].min()) / (df['Data'].max() - df['Data'].min())

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(index=train_df.index)

# reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# pytorch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataloaders
train_set = CustomDataset(train_df)
val_set = CustomDataset(val_df)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

# load activity predictor model
model = FFNN(input_size, output_size, hidden_layers)
model.to(device)

# Define the loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# training loop
def train(model, train_loader, val_loader, loss_fn, optimizer, device, epochs, save_path):
    best_val_loss = float('inf')
    train_loss, val_loss = [], []
    for epoch in range(epochs):
        model.train()
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

            running_loss += loss.item() * batch_size

        epoch_loss = running_loss / len(train_loader)
        train_loss.append(epoch_loss / len(train_loader))

        val_loss, val_rmse, val_pearson = validate(model, val_loader, loss_fn)
        with open('train_log', 'a') as f:
            print(f"Epoch {epoch + 1}:: train loss: {epoch_loss:.4f}, val loss: {val_loss:.4f}, val RMSE: {val_rmse}, val pearson: {val_pearson}", file=f)

        val_loss.append(val_loss / len(val_loader))
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), os.path.join(save_path, 'activity_model'))
            with open('train_log', 'a') as f:
                print('Saved best model', file=f)
            best_val_loss = val_loss

train(model, train_loader, val_loader, loss_fn, optimizer, device, epochs, save_path)
