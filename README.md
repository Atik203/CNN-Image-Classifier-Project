# Plant Disease Image Classification with CNN

Mini CNN project for classifying plant leaf diseases using the PlantVillage dataset from Kaggle.

Dataset: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage)

## Project Overview

This project trains a Convolutional Neural Network (CNN) in PyTorch to classify plant leaf images into disease categories.

- Framework: PyTorch
- Dataset source: Kaggle (`mohitsingh1804/plantvillage`)
- Number of classes in dataset: 38
- Image size used: `128 x 128`
- Notebook: `cnn_project.ipynb`

## Model Architecture

The model (`MyCNN`) has:

- 3 convolution blocks (`Conv2d -> ReLU -> BatchNorm -> MaxPool`)
- A fully connected classifier with dropout
- Final linear layer outputting `num_classes` logits

Classifier pipeline:

- `Flatten`
- `Linear(128*16*16 -> 128)` + ReLU + Dropout(0.4)
- `Linear(128 -> 64)` + ReLU + Dropout(0.4)
- `Linear(64 -> num_classes)`

## Data Pipeline

- Images are loaded from:
  - `PlantVillage/train`
  - `PlantVillage/val`
- Custom dataset class: `MultiClassClassfication`
- Transformations:
  - Resize to `128x128`
  - Convert to tensor
  - Normalize with mean/std = `[0.5, 0.5, 0.5]`

## Training Setup

- Optimizer: `Adam`
- Learning rate: `0.001`
- Loss: `CrossEntropyLoss`
- Epochs: `10`
- Batch size: `32`
- Device: CUDA if available, otherwise CPU

## Current Results (from notebook run)

- Full train size: `43444`
- Full validation size: `10861`
- Logged test accuracy: `1.0`

Important: the notebook currently trains and evaluates on a small subset (`100` train and `100` val images) for quick testing:

```python
train_dataset = Subset(train_dataset_full, list(range(min(100, len(train_dataset_full)))))
test_dataset = Subset(test_dataset_full, list(range(min(100, len(test_dataset_full)))))
```

Because of this, the reported accuracy is not a realistic measure of full-dataset performance.

## How to Run

1. Open `cnn_project.ipynb`.
2. Install dependencies (example):
   - `torch`
   - `torchvision`
   - `pillow`
   - `kagglehub`
3. Run notebook cells in order:
   - Download dataset
   - Build dataset/dataloaders
   - Initialize model
   - Train
   - Evaluate

## Suggested Next Improvements

- Train on the full train/val split instead of the 100-image subset.
- Add train/validation loss and accuracy plots.
- Save and load model checkpoints (`.pth`).
- Add single-image prediction/inference function.
- Compute a confusion matrix and per-class accuracy.

## License and Data Usage

Please follow the dataset license and usage terms provided on the Kaggle dataset page.
