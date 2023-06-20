import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim, dropout):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            linear_layer = nn.Linear(last_dim, hidden_dim)
            nn.init.kaiming_normal_(linear_layer.weight)
            self.layers.append(linear_layer)
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.mu = nn.Linear(last_dim, z_dim)
        nn.init.kaiming_normal_(self.mu.weight)
        self.var = nn.Linear(last_dim, z_dim)
        nn.init.kaiming_normal_(self.var.weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        z_mu = self.mu(x)
        z_var = self.var(x)
        return z_mu, z_var


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dims, output_dim, dropout):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        last_dim = z_dim
        for hidden_dim in reversed(hidden_dims):
            linear_layer = nn.Linear(last_dim, hidden_dim)
            nn.init.kaiming_normal_(linear_layer.weight)
            self.layers.append(linear_layer)
            self.layers.append(nn.Dropout(dropout))
            self.layers.append(nn.ReLU())
            last_dim = hidden_dim
        self.out = nn.Linear(last_dim, output_dim)
        nn.init.kaiming_normal_(self.out.weight)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        predicted = torch.sigmoid(self.out(x))
        return predicted


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, z_dim, dropout=0.0):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dims, z_dim, dropout)
        self.decoder = Decoder(z_dim, hidden_dims, input_dim, dropout)

    def forward(self, x):
        z_mu, z_var = self.encoder(x)
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)
        predicted = self.decoder(x_sample)
        return predicted, z_mu, z_var