import torch
import torch.nn as nn

# Define the network architecture
class FFNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super().__init__()

        # Create input and output layers
        self.input_layer = nn.Linear(input_size, hidden_layers[0])
        self.output_layer = nn.Linear(hidden_layers[-1], output_size)

        # Create hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            layer = nn.Linear(hidden_layers[i], hidden_layers[i + 1])
            self.hidden_layers.append(layer)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = nn.functional.relu(x)
        x = self.output_layer(x)
        return x
