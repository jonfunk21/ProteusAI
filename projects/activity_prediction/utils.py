from torch.utils.data import Dataset

class VAEDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.data['label'].iloc[index]
        x = self.data['x'].iloc[index]
        return x