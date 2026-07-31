import torch
from torch.utils.data import Dataset

class XNLIDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=128):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset[idx]
        premise = item['premise']
        hypothesis = item['hypothesis']
        label = item['label']
        
        encoding = self.tokenizer(
            premise,
            hypothesis,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }
