from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().strip().split('\n')
        self.examples = lines
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]
        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_tensors="pt")
        # For language modeling, inputs and labels are the same shifted by one token - 
        # but for simplicity, assume we're doing masked language modeling or something similar.
        return {k: v.squeeze() for k,v in encoding.items()}