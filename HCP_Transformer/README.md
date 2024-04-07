# Transformer Model for Predicting Gender

This folder contains files that apply a Transformer-based neural network to predict cognitive scores from brain connectivity data. It includes a Transformer model and utility scripts for data preparation, training, testing, and result aggregation.

## Project Structure

- `transformer_model.py`: Defines the Transformer neural network architecture and the MLP block used for regression tasks.
- `utils.py`: Contains utility functions for data preprocessing, including scaling and custom dataset creation for PyTorch.
- `train_test_functs.py`: Functions for training the model with reconstruction loss, testing the model, and saving predictions.
- `main_trans.py`: Main script to execute the training and testing of the Transformer model using provided brain connectivity data.
- `create_Y_traintest.py`: Script to create training and testing datasets for cognitive scores.
- `overall_results.py`: Aggregates results from different runs to provide an overview of the model performance.

## Usage

To run the main Transformer model execute:

```bash
python main_trans.py
```

## Data directory
- FC dataset: Located in /HCP_FC/X.npy
- SC dataset: Located in /HCP_SC/X.npy
- Scores: Located in /scores, one for each score