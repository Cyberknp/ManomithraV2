# ManomithraV2 Models

This directory contains the machine learning components for the facial expression recognition (emotion detection) module of ManomithraV2. It includes the scripts for data preparation, model training, and quality evaluation, along with the saved model artifacts.

## Directory Structure

- **`facedetection.py`**: The main script for building and training the lightweight CNN architecture for emotion detection. It loads images from the `facial/` directories, sets up data augmentation, builds a multi-block separable convolution network, and trains the model utilizing `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint` callbacks.
- **`preparation.py`**: A utility script providing an `ImageDataGenerator` setup for standardizing and augmenting training images.
- **`quality_review.py`**: An extensive evaluation script that calculates test metrics (loss, accuracy, precision, recall, F1-scores) for the generated models. It performs side-by-side model comparisons, generates detailed per-class classification reports, and saves the output (as JSON metrics, bar charts, and confusion matrix plots).
- **`reports/`**: The output directory populated by the `quality_review.py` script, storing JSON metrics files (e.g., `best_metrics.json`, `emotion_metrics.json`, `model_comparison.json`) and evaluation plots.
- **`best_model.keras`**: A Keras model checkpoint saved during training representing the epoch with the highest validation accuracy.
- **`emotion_model.keras`**: The final Keras model saved at the completion of the training run.

## Usage

### 1. Training the Model
To train the facial expression detection model from scratch, execute:
```bash
python facedetection.py
```
This script reads images from the `facial/train/` data folder and will output the trained models (`best_model.keras` and `emotion_model.keras`) to this directory.

### 2. Evaluating Model Quality
To compute test performance and generate quality reports on the test dataset:
```bash
python quality_review.py
```
*Optional command-line arguments:*
- `--model best` or `--model emotion` to evaluate only a specific model.
- `--no-plots` to skip generating Matplotlib graphs (confusion matrix, bar charts).
- `--history path/to/history.json` to plot loss and accuracy curves from training.

Quality reports and visualisations will be written to the `reports/` folder.
