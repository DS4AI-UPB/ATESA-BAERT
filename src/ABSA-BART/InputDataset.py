import torch
from torch.utils.data import Dataset

class InputDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.data = df
        self.text = df.text
        self.tokenizer = tokenizer
        self.targets = self.data.absa_tags

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        tokens = self.data['tokens'].iloc[idx]
        tags = self.data['absa_tags'].iloc[idx]

        data_tokens = []
        data_tags = []

        for i in range(len(tokens)):
            t = self.tokenizer.tokenize(tokens[i])
            data_tokens += t
            data_tags += [int(tags[i])] * len(t)
        
        data_ids = self.tokenizer.convert_tokens_to_ids(data_tokens)

        ids_tensor = torch.tensor(data_ids)
        tags_tensor = torch.tensor(data_tags)

        return data_tokens, ids_tensor, tags_tensor
    
    def __len__(self):
        return len(self.data)