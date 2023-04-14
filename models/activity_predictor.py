import torch.nn as nn

class FFNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, dropout_rate):
        super(FFNN, self).__init__()
        self.layers = []
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))

        self.layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)