import torch
from torch import nn

class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, dropout_p):
        super(FFNN, self).__init__()
        # Define the layers
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_p))

        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_p))

        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))
        self.layers.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
