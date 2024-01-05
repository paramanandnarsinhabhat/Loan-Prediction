# Loan Prediction Model

This project aims to predict loan approval outcomes based on several input features using a neural network implemented in Keras.

## Pre-processing Steps

The pre-processing script follows these steps:
1. Filling the missing values with mode for categorical variables and mean for continuous variables.
2. Converting categorical variables to numerical representations using mapping.
3. Scaling all variables to a range of 0 to 1 to normalize the data.

## Installation

Before running the pre-processing and model scripts, ensure the following libraries are installed:

```
pip install tensorflow matplotlib scikit-learn numpy pandas
```

## Usage

Run the `loan_prediction_preprocessing.py` script to preprocess the data. The preprocessed data will be saved for model training. After pre-processing, execute the `loan_prediction_nueral_network_keras.py` to train the neural network model and evaluate its performance.

## File Structure

- `data/`
  - `loan_data.csv`: The original dataset.
  - `after_preprocessing/`: Contains processed data after running the pre-processing script.
- `scripts/`
  - `loan_prediction_nueral_network_keras.py`: The neural network model script.
  - `loan_prediction_preprocessing.py`: The data pre-processing script.
- `requirements.txt`: Lists all the dependencies for the project.

## Model Training and Evaluation

The model is trained on pre-processed data with a defined architecture, using binary cross-entropy loss and the Adam optimizer. Performance is visualized by plotting accuracy and loss over epochs for both training and validation sets.

## Output

After training, the model's accuracy on the validation set is printed out, along with visualizations of the model's loss and accuracy over training epochs.

