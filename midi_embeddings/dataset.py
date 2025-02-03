import json
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer


class MIDIDatasetDynamic(Dataset):
    def __init__(
        self,
        max_seq_len: int,
        split: str = "train",
        tokenizer_path: str = None,
        limit: int = None,
        return_info: bool = False,
    ):
        """MAESTRO dataset for MIDI embeddings

        Args:
            max_seq_len: Maximum sequence length for truncation/padding
            split: Dataset split (train/test/validation)
            tokenizer_path: Path to pre-trained tokenizer config. If None, train tokenizer
            limit: Optional limit for quick testing
            return_info: return additional information about each sample
        """
        self.max_seq_len = max_seq_len
        self.return_info = return_info

        # Load dataset
        if split == "all":
            train_ds = load_dataset("epr-labs/maestro-sustain-v2", split="train")
            val_ds = load_dataset("epr-labs/maestro-sustain-v2", split="validation")
            test_ds = load_dataset("epr-labs/maestro-sustain-v2", split="test")
            self.dataset = concatenate_datasets([train_ds, val_ds, test_ds])
        else:
            self.dataset = load_dataset("epr-labs/maestro-sustain-v2", split=split)

        if limit:
            self.dataset = self.dataset.shuffle(seed=42).select(range(limit))

        # Initialize tokenizer
        base_tokenizer = ExponentialTimeTokenizer()

        if tokenizer_path is None:
            tokenizer_path = "awesome.json"
            tokenizer = AwesomeMidiTokenizer(base_tokenizer=base_tokenizer)
            tokenizer.train(self.dataset)
            tokenizer.save_tokenizer(tokenizer_path)

        self.tokenizer = AwesomeMidiTokenizer(base_tokenizer=base_tokenizer).from_file(tokenizer_path)

        # Preprocess dataset
        self.dataset = self.dataset.map(
            self._preprocess_example,
            desc=f"Preprocessing {split} dataset...",
            load_from_cache_file=True,
            num_proc=4,
        )

    def __len__(self):
        return len(self.dataset)

    def _preprocess_example(self, example):
        # Tokenize notes
        notes = pd.DataFrame(example["notes"])
        tokens = self.tokenizer.tokenize(notes)
        token_ids = self.tokenizer.awesome_tokens_to_base_ids(tokens)

        # Truncate/pad sequence
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[: self.max_seq_len]
        else:
            padding = [0] * (self.max_seq_len - len(token_ids))
            token_ids += padding

        return {
            "token_ids": token_ids,
        }

    def __getitem__(self, idx):
        example = self.dataset[idx]
        token_ids = torch.tensor(example["token_ids"], dtype=torch.int64)
        info_dict = json.loads(example["source"])

        if self.return_info:
            return {
                "token_ids": token_ids,
                "info": info_dict,
            }
        else:
            return token_ids

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size


class MIDIDatasetPresaved(Dataset):
    def __init__(
        self,
        max_seq_len: int,
        split: str = "train",
        tokenizer_path: str = None,
        limit: int = None,
        return_info: bool = False,
        force_retokenize: bool = False,
    ):
        """Dataset that saves tokenized data for faster subsequent loads

        Args:
            max_seq_len: Maximum sequence length for truncation/padding
            split: Dataset split (train/test/validation)
            tokenizer_path: Path to pre-trained tokenizer config. If None, train tokenizer
            limit: Optional limit for quick testing
            return_info: return additional information about each sample
            force_retokenize: Force re-tokenization of the dataset
        """
        self.max_seq_len = max_seq_len
        self.return_info = return_info
        self.save_path = f"data/{split}_tokenized_{max_seq_len}.json"

        if not Path(self.save_path).parent.exists():
            Path(self.save_path).parent.mkdir(parents=True)

        # Try loading pre-tokenized data
        if not force_retokenize and Path(self.save_path).exists():
            print(f"Loading pre-tokenized data from {self.save_path}")
            self._load_tokenized()
            if limit:
                self.dataset = self.dataset[:limit]
            return

        # Load and process original dataset
        self._load_and_process_original(split, limit, tokenizer_path)
        self._save_tokenized()
        self._load_tokenized()

    def _load_and_process_original(self, split, limit, tokenizer_path):
        """Load and tokenize original dataset"""
        # Dataset loading
        if split == "all":
            train = load_dataset("epr-labs/maestro-sustain-v2", split="train")
            val = load_dataset("epr-labs/maestro-sustain-v2", split="validation")
            test = load_dataset("epr-labs/maestro-sustain-v2", split="test")
            self.dataset = concatenate_datasets([train, val, test])
        else:
            self.dataset = load_dataset("epr-labs/maestro-sustain-v2", split=split)

        if limit:
            self.dataset = self.dataset.shuffle(seed=42).select(range(limit))

        # Tokenizer setup
        base_tokenizer = ExponentialTimeTokenizer()
        if tokenizer_path is None:
            self.tokenizer = AwesomeMidiTokenizer(base_tokenizer=base_tokenizer)
            self.tokenizer.train(self.dataset)
            self.tokenizer.save_tokenizer("awesome.json")
        else:
            self.tokenizer = AwesomeMidiTokenizer(base_tokenizer=base_tokenizer).from_file(tokenizer_path)

        # Tokenization
        original_columns = self.dataset.column_names
        self.dataset = self.dataset.map(
            self._tokenize_example,
            remove_columns=[col for col in original_columns if col != "source"],
            num_proc=4,
            desc="Tokenizing...",
            load_from_cache_file=False,
        )

    def _tokenize_example(self, example):
        """Tokenize single example"""
        try:
            # Convert notes and tokenize
            notes_df = pd.DataFrame(example["notes"])
            tokens = self.tokenizer.tokenize(notes_df)
            token_ids = self.tokenizer.awesome_tokens_to_base_ids(tokens)

            # Pad/truncate sequence
            seq_len = len(token_ids)
            if seq_len > self.max_seq_len:
                token_ids = token_ids[: self.max_seq_len]
            else:
                token_ids += [0] * (self.max_seq_len - seq_len)

            return {"token_ids": token_ids, "source": example["source"]}
        except Exception as e:
            print(f"Error processing example: {e}")
            return {"token_ids": [], "source": ""}

    def _save_tokenized(self):
        """Save processed data to JSON"""
        valid_data = []
        for ex in self.dataset:
            if len(ex["token_ids"]) == self.max_seq_len and ex["source"]:
                valid_data.append({"token_ids": ex["token_ids"], "source": ex["source"]})

        with open(self.save_path, "w") as f:
            json.dump(valid_data, f, indent=2)

    def _load_tokenized(self):
        """Load pre-tokenized data from JSON"""
        with open(self.save_path) as f:
            self.dataset = json.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        token_ids = torch.tensor(example["token_ids"], dtype=torch.int64)

        if not self.return_info:
            return token_ids

        try:
            info = json.loads(example["source"])
        except json.JSONDecodeError:
            info = {"error": "Invalid source info"}

        return {"token_ids": token_ids, "info": info}

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size


if __name__ == "__main__":
    # Test dynamic dataset
    dataset = MIDIDatasetDynamic(
        max_seq_len=1024,
        split="train",
        limit=100,
        return_info=True,
        tokenizer_path="awesome.json",
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Example: {dataset[0]}")

    print("DataLoader test:")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch['token_ids'].shape}")

    # Test pre-saved dataset
    print("File dataset test:")
    dataset = MIDIDatasetPresaved(
        max_seq_len=2048,
        split="train",
        return_info=True,
        tokenizer_path="awesome.json",
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Example: {dataset[0]}")

    print("DataLoader test:")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    print(f"Batch shape: {batch['token_ids'].shape}")
