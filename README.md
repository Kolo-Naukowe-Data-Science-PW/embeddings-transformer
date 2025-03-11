# MIDI Embeddings

## Overview

This project implements a Transformer-based model for generating embeddings from MIDI files, focusing on learning meaningful representations of musical pieces.

### W&B
`https://www.wandb.ai/knds-midi/midi-embeddings`

## Main Script: `midi_embeddings.py`

### Purpose
The `midi_embeddings.py` script serves as the primary entry point for training and evaluating a MIDI embedding model using Hydra for configuration management. When a user runs this script, the following will happen:

### Workflow
1. **Configuration Setup**
   - Loads configuration from YAML files in the `configs` directory using Hydra
   - Parameters include sequence length, embedding dimensions, attention heads, model layers, batch size, epochs, learning rate, and dropout rate

2. **Dataset Preparation**
   - Loads three datasets:
     - Training dataset
     - Validation dataset
     - Visualization dataset
   - Uses `MIDIDatasetPresaved` and `MIDIDatasetDynamic` for efficient data handling
   - Tokenizes MIDI files using a pre-trained tokenizer if available, otherwise trains a new one

3. **Model Initialization**
   - Creates a `MIDITransformerEncoder` with the specified configuration

4. **Model Training**
   - Trains the model using the training dataset
   - Validates performance on the validation dataset
   - Saves the best-performing model checkpoint
   - Optionally logs training metrics and embedding visualizations to Weights & Biases

5. **Model Evaluation**
   - Evaluates the trained model on the test dataset
   - Prints out the test loss and perplexity metrics

6. **Embedding Visualization**
   - Generates and saves an interactive HTML visualization of embeddings using t-SNE
   - Creates animations showing how embeddings evolve during training

### Example Usage
```bash
# Run with default configuration
python midi_embeddings.py

# Override specific parameters
python midi_embeddings.py train.batch_size=16 train.epochs=20

# Use a different configuration file
python midi_embeddings.py -cn another_config_file
```

## Evaluation mode
For evaluation mode, run the script with `eval.yaml` config file as shown below:
```bash
python midi_embeddings.py -cn eval eval.model_path=path/to/your/model.pth
```
This lets the user use an already trained model to evaluate on the test dataset and visualize the embeddings.

## Installation
```bash
pip install -r requirements.txt
```

## Key Components
- `transformer.py`: Defines the MIDI Transformer Encoder architecture
- `dataset.py`: Handles MIDI dataset loading and preprocessing
- `train.py`: Contains training and evaluation functions
- `visualize.py`: Provides embedding visualization functions

## Configuration
The project uses Hydra for configuration management. Configuration files are located in the configs directory:

- `config.yaml`: Base configuration
- Additional configuration files for different experiments

## Sample Configuration Structure
```yaml
# Default configuration

train:
  max_seq_len: 2048
  embed_dim: 384
  nhead: 6
  num_layers: 4
  batch_size: 8
  epochs: 50
  learning_rate: 3e-4
  weight_decay: 0.01
  dropout: 0.2
  model_name: "embed_model"

dataset:
  tokenizer_path: "awesome.json"
  limit: null

wandb:
  project: "midi-embeddings"
  viz_interval: 2

device: "cuda"
seed: 42
```

## Visualization
The script generates an interactive HTML visualization of song embeddings, allowing users to explore how different musical pieces are represented in the embedding space. It can also create animations showing how embeddings evolve during training.
