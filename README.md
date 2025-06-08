
# PneumoniaMNIST Classification with ResNet50

This repo contains code to train and evaluate a ResNet50 model on the PneumoniaMNIST dataset.

## Files

- `dataset.py` : Dataset class for loading PneumoniaMNIST images.
- `train.py` : Script to train the model.
- `evaluate.py` : Script to evaluate the model on validation data.
- `requirements.txt` : Python dependencies.

## Usage

1. Prepare data (convert images and labels to `.npy` files).
2. Train the model:
3. Evaluate the model:

## Hyperparameters

- Learning rate: 0.001
- Batch size: 32
- Epochs: 5
- Optimizer: Adam
- Loss function: CrossEntropyLoss

Adjust hyperparameters in `train.py` as needed.
