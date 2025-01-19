import json

import pandas as pd
from fortepyan import MidiPiece
from datasets import load_dataset
from midi_tokenizers import AwesomeMidiTokenizer, ExponentialTimeTokenizer


def maestro_download():
    dataset = load_dataset("epr-labs/maestro-sustain-v2", split="train")

    base = ExponentialTimeTokenizer()
    awesome = AwesomeMidiTokenizer(base_tokenizer=base).from_file("awesome.json")

    tokens_dict = {idx: {} for idx in range(0, 962)}

    for record_idx in range(0, 962):
        record = dataset[record_idx]

        piece = MidiPiece.from_huggingface(record)
        piece_df = pd.DataFrame(piece.df)

        tokens = awesome.tokenize(piece_df)

        tokens_dict[record_idx]["tokens"] = tokens
        tokens_dict[record_idx]["info"] = piece.source

    with open("maestro_tokens.json", "w") as f:
        json.dump(tokens_dict, f)
