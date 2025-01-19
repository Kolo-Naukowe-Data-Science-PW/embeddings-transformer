from torch.utils.data import Dataset
import json
import torch
from midi_tokenizers import ExponentialTimeTokenizer, AwesomeMidiTokenizer

class MIDIDataset(Dataset):
    def __init__(self, tokens_path, max_seq_len, indices=None):
        self.max_seq_len = max_seq_len
        
        base_tokenizer = ExponentialTimeTokenizer()
        self.tokenizer = AwesomeMidiTokenizer(base_tokenizer=base_tokenizer).from_file("awesome.json")
        
        # Load data
        with open(tokens_path) as f:
            all_tokens = json.load(f)
            
        if indices is None:
            indices = range(len(all_tokens))
        
        # Extract specified range of songs
        self.data = {
            "tokens": [all_tokens[str(idx)]["tokens"] for idx in indices],
            "info": [all_tokens[str(idx)]["info"] for idx in indices]
        }
        
        self.tokenized_songs = []
        for song in self.data["tokens"]:
            token_ids = self.tokenizer.awesome_tokens_to_base_ids(song)
            self.tokenized_songs.append(token_ids)
    
    def __len__(self):
        return len(self.tokenized_songs)
    
    def __getitem__(self, idx):
        token_ids = self.tokenized_songs[idx]
        
        # Handle sequence length
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            # Pad if song is too short
            padding = [0] * (self.max_seq_len - len(token_ids))
            token_ids.extend(padding)
        
        
        return torch.tensor(token_ids, dtype=torch.int64)
    
    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size
    
    
if __name__ == "__main__":
    dataset = MIDIDataset("tokens.json", 0, 962, 1024)
    
    print(dataset[0])
