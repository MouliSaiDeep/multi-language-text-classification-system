import pytest
import torch
from datasets import Dataset
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import XNLIDataset
from src.data.sampler import BalancedBatchSampler

class MockTokenizer:
    def __call__(self, premise, hypothesis, padding=None, truncation=None, max_length=None, return_tensors=None):
        return {
            "input_ids": torch.ones((1, max_length or 128), dtype=torch.long),
            "attention_mask": torch.ones((1, max_length or 128), dtype=torch.long)
        }

def test_data_and_sampler():
    # Build a tiny fake HF-style dataset with 3 languages (4 samples each)
    data = {
        "premise": [f"premise_{i}" for i in range(12)],
        "hypothesis": [f"hypothesis_{i}" for i in range(12)],
        "label": [i % 3 for i in range(12)],
        "language": ["en", "en", "en", "en", "fr", "fr", "fr", "fr", "es", "es", "es", "es"]
    }
    hf_dataset = Dataset.from_dict(data)
    
    tokenizer = MockTokenizer()
    dataset = XNLIDataset(hf_dataset, tokenizer, max_length=64)
    
    # Assert dataset basic functionality
    assert len(dataset) == 12
    item = dataset[0]
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "label" in item
    assert item["input_ids"].shape == (64,)
    assert item["attention_mask"].shape == (64,)
    assert item["label"].item() == 0

    # Build BalancedBatchSampler with batch_size=3
    sampler = BalancedBatchSampler(dataset, batch_size=3, drop_last=True)
    
    # 12 samples total, batch_size=3, we expect 4 batches
    assert len(sampler) == 4
    
    batches = list(sampler)
    assert len(batches) == 4
    
    for batch in batches:
        assert len(batch) == 3
        # Check that batch contains more than one distinct language
        batch_langs = [dataset.hf_dataset["language"][idx] for idx in batch]
        # With 3 languages and batch_size 3, each batch should contain exactly one of each
        assert set(batch_langs) == {"en", "fr", "es"}
