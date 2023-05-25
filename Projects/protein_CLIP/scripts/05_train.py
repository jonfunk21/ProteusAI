from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn


script_path = '.'
data_dir = os.path.join(script_path, '../data')
df_path = os.path.join(data_dir, 'processed/04_dataset.csv')
fasta_dir = os.path.join(data_dir, 'fastas')

df = pd.read_csv(df_path)

keep = [f[:-3] for f in os.listdir(os.path.join(data_dir, 'embeddings/proteins')) if f.endswith('.pt')]
df = df[df['protein'].isin(keep)]
keep = [f[3:-3].replace('_','.') for f in os.listdir(os.path.join(data_dir, 'embeddings/descriptions')) if f.endswith('.pt')]
df = df[df['EC'].isin(keep)]

# Split the data into train/val and test datasets
train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(index=train_df.index)

# Reset indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

batch_size = 2*10

class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        n = self.data['protein'].iloc[index]
        ec = self.data['EC'].iloc[index]
        x = torch.load(f'../data/embeddings/proteins/{n}.pt', map_location=self.device)
        y = torch.load(f'../data/embeddings/descriptions/EC_{ec.replace(".","_")}.pt', map_location=self.device)
        return x, y

train_data = CustomDataset(train_df)
val_data = CustomDataset(val_df)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)


class CFG:
    debug = False
    protein_path = "C:/Moein/AI/Datasets/Flicker-8k/Images"
    captions_path = "C:/Moein/AI/Datasets/Flicker-8k"
    batch_size = 32
    num_workers = 4
    head_lr = 1e-3
    protein_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    protein_embedding = 1280
    text_encoder_model = "ESM 1v"
    text_embedding = 1280
    text_tokenizer = "GPT2-large"
    max_length = 200

    pretrained = True  # for both protein encoder and text encoder
    trainable = True  # for both protein encoder and text encoder
    temperature = 1.0

    # for projection head; used for both protein and text encoders
    num_projection_layers = 1
    projection_dim = 256
    dropout = 0.1


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class ProjectionHead(nn.Module):
    def __init__(
            self,
            embedding_dim,
            projection_dim=CFG.projection_dim,
            dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class CLIPModel(nn.Module):
    def __init__(
            self,
            temperature=CFG.temperature,
            protein_embedding=CFG.protein_embedding,
            text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.protein_projection = ProjectionHead(embedding_dim=protein_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, protein_features, text_features):
        # Getting Protein and Text Embeddings (with same dimension)
        protein_embeddings = self.protein_projection(protein_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ protein_embeddings.T) / self.temperature
        protein_similarity = protein_embeddings @ protein_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (protein_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        protein_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss = (protein_loss + texts_loss) / 2.0  # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


# train
def train_epoch(model, train_loader, optimizer, lr_scheduler):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        x = batch[0]
        y = batch[1]
        loss = model(x,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count = x.size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    return loss_meter


def val_epoch(model, val_loader):
    loss_meter = AvgMeter()

    tqdm_object = tqdm(val_loader, total=len(val_loader))
    for batch in tqdm_object:
        x = batch[0]
        y = batch[1]
        loss = model(x,y)

        count = x.size(0)
        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(val_loss=loss_meter.avg)
    return loss_meter



model = CLIPModel().to(CFG.device)
params = [
    {"params": itertools.chain(
        model.protein_projection.parameters(), model.text_projection.parameters()
    ), "lr": CFG.head_lr, "weight_decay": CFG.weight_decay}
]
optimizer = torch.optim.AdamW(params, weight_decay=0.)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=CFG.patience, factor=CFG.factor
)

best_loss = float('inf')
for epoch in range(CFG.epochs):
    print(f"Epoch: {epoch + 1}")
    model.train()
    train_loss = train_epoch(model, train_loader, optimizer, lr_scheduler)
    model.eval()
    with torch.no_grad():
        val_loss = val_epoch(model, val_loader)

    if val_loss.avg < best_loss:
        best_loss = val_loss.avg
        torch.save(model.state_dict(), "model.pt")
        print("Saved Best Model!")

    lr_scheduler.step(val_loss.avg)