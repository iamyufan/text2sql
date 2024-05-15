from torch.utils.data import Dataset


class SQLDataset(Dataset):
    def __init__(self, texts, queries=None):
        self.texts = texts
        self.queries = queries

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.queries is not None:
            return self.texts[idx], self.queries[idx]
        return self.texts[idx], None
