# MIDI Embeddings

## Overview

This project implements a Transformer-based model for generating embeddings from MIDI files, focusing on learning meaningful representations of musical pieces.

## Main Script: `main.py`

### Purpose
The `main.py` script serves as the primary entry point for training and evaluating a MIDI embedding model. When a user runs this script, the following will happen:

### Workflow
1. **Configuration Setup**
   - A configuration dictionary is created with hyperparameters for the model and training process
   - Parameters include sequence length, embedding dimensions, attention heads, model layers, batch size, epochs, learning rate, and dropout rate

2. **Dataset Preparation**
   - Loads three datasets:
     - Training dataset
     - Validation dataset
     - Test dataset
   - Uses `MIDIDatasetPresaved` and `MIDIDatasetDynamic` for efficient data handling
   - Tokenizes MIDI files using a pre-trained tokenizer if available, otherwise trains a new one

3. **Model Initialization**
   - Creates a `MIDITransformerEncoder` with the specified configuration

4. **Model Training**
   - Trains the model using the training dataset
   - Validates performance on the validation dataset
   - Saves the best-performing model checkpoint

5. **Model Evaluation**
   - Evaluates the trained model on the test dataset
   - Prints out the test loss and perplexity metrics

6. **Embedding Visualization**
   - Generates and saves an interactive HTML visualization of embeddings for all the songs in the MAESTRO-sustain-v2 dataset using t-SNE

### Example Usage
```bash
python main.py
```

## Installation
```bash
pip install requirements.txt
```

## Key Components
- `transformer.py`: Defines the MIDI Transformer Encoder architecture
- `dataset.py`: Handles MIDI dataset loading and preprocessing
- `train.py`: Contains training and evaluation functions
- `visualize.py`: Provides embedding visualization functions

## Customization
Users can modify the configuration dictionary in `main.py` to experiment with different hyperparameters, such as:
- Embedding dimensions
- Number of attention heads
- Number of model layers
- Learning rate
- Batch size
- Dropout rate
- Sequence length
- Number of epochs

## Visualization
The script generates an interactive HTML visualization of song embeddings, allowing users to explore how different musical pieces are represented in the embedding space.
