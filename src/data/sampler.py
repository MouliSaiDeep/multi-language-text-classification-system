import numpy as np
import torch
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by language column in underlying HF dataset
        self.lang_to_indices = {}
        languages = self.dataset.hf_dataset["language"]
        for idx, lang in enumerate(languages):
            if lang not in self.lang_to_indices:
                self.lang_to_indices[lang] = []
            self.lang_to_indices[lang].append(idx)
            
        self.langs = list(self.lang_to_indices.keys())
        self.num_langs = len(self.langs)
        
        if self.num_langs == 0:
            raise ValueError("Dataset has no language information associated with it.")
            
        # Determine base allocation and remainder
        self.base_alloc = self.batch_size // self.num_langs
        self.remainder = self.batch_size % self.num_langs
        
        # Calculate exactly how many batches can be formed
        self.num_batches = self._calculate_num_batches()

    def _calculate_num_batches(self):
        sizes = {lang: len(indices) for lang, indices in self.lang_to_indices.items()}
        count = 0
        remainder_cycle = 0
        
        while True:
            # Determine allocation for this simulated batch
            alloc = {lang: self.base_alloc for lang in self.langs}
            for i in range(self.remainder):
                lang = self.langs[(remainder_cycle + i) % self.num_langs]
                alloc[lang] += 1
                
            # Check if all language pools have enough samples
            possible = True
            for lang in self.langs:
                if sizes[lang] < alloc[lang]:
                    possible = False
                    break
                    
            if not possible:
                break
                
            # Deduct the samples
            for lang in self.langs:
                sizes[lang] -= alloc[lang]
                
            count += 1
            remainder_cycle = (remainder_cycle + self.remainder) % self.num_langs
            
        return count

    def __iter__(self):
        # Reshuffle pools at start of each epoch/iteration
        pools = {lang: np.random.permutation(indices).tolist() 
                 for lang, indices in self.lang_to_indices.items()}
                 
        remainder_cycle = 0
        for _ in range(self.num_batches):
            batch = []
            # Determine allocation for this batch
            alloc = {lang: self.base_alloc for lang in self.langs}
            for i in range(self.remainder):
                lang = self.langs[(remainder_cycle + i) % self.num_langs]
                alloc[lang] += 1
                
            # Extract samples from language pools
            for lang in self.langs:
                for _ in range(alloc[lang]):
                    batch.append(pools[lang].pop())
                    
            remainder_cycle = (remainder_cycle + self.remainder) % self.num_langs
            # Shuffle internal batch elements to mix languages
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches
