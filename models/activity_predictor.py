import torch
import torch.nn as nn
import math

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x

class AttentionModel(nn.Module):
    def __init__(self, input_size, sequence_length, d_model, nhead, num_layers, dropout_rate, output_size=1):
        super(AttentionModel, self).__init__()

        self.embedding = nn.Linear(input_size, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.pos_encoder = PositionalEncoding(d_model, sequence_length)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout_rate),
            num_layers=num_layers
        )
        self.fc = nn.Linear(d_model * sequence_length, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.pos_encoder(x.permute(1, 0, 2))
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2).reshape(x.shape[1], -1)
        x = self.fc(x)
        return x

    def get_attentions_and_embeddings(self, x):
        # Compute the input embeddings
        embeddings = self.embedding(x)

        # Initialize a list to store attention weights
        attentions = []

        # Pass the embeddings through each layer of the transformer, saving attention weights
        for layer in self.transformer_layers:
            embeddings, attn = layer(embeddings)
            attentions.append(attn)

        # Pass the final embeddings through the output layer
        output = self.output_layer(embeddings)

        return attentions, embeddings, output