from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tabular_data import load_airbnb
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter 
import datetime
import joblib
import json
import math
import numpy as np
import pandas as pd
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.model_selection import KFold

# Set the random seed for reproducibility
np.random.seed(1)

# Define the path to the data file
file_path = '/Users/andreasyianni/Desktop/Educational Training /Data Scientist Training/AiCore Training/Projects/Modelling-Airbnbs-Property-Listing-Dataset/airbnb-property-listings/tabular_data/clean_tabular_data.csv'

data_nn = load_airbnb(file_path, "Price_Night") # Load the dataset

df = pd.DataFrame(data_nn[0])  
missing_values = df.isnull().sum()
print(missing_values)

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(data_nn[0])  # Standardize the features (X)

# Create a tuple containing the standardized features and labels
data_nn_standardized = (X_standardized, data_nn[1])

class AirbnbNightlyPriceRegressionDataset(Dataset):
    """
    PyTorch Dataset for Airbnb nightly price regression.

    This dataset class is designed for regression tasks and provides access to
    features and labels for predicting Airbnb nightly prices.

    Attributes:
    -----------
    features (torch.Tensor): The input features as a float tensor.
    labels (torch.Tensor): The corresponding labels as a float tensor.

    Methods:
    --------
    __init__(self, data_nn):
        - Initialize the Airbnb nightly price regression dataset.

    __getitem__(self, idx):
        - Get a specific data sample by index.
        
    __len__(self):
        - Get the total number of data samples in the dataset.
    """


    def __init__(self, data_nn): 
        """
        Initialize the Airbnb nightly price regression dataset.

        Parameters:
        -----------
             - data_nn (tuple): A tuple containing data for features and labels as numpy arrays.
               The first element of the tuple represents features, and the second element represents labels.
        """
        super().__init__()
        self.features = torch.from_numpy(np.array(data_nn[0])).float()
        self.labels = torch.from_numpy(np.array(data_nn[1])).float()

    def __getitem__(self, idx):
        """
        Get a specific data sample by index.

        Parameters:
        -----------
            - idx (int): Index of the data sample to retrieve.

        Returns:
        --------
            - tuple: A tuple containing features and labels for the specified index.
        """
        features = self.features[idx]
        labels = self.labels[idx]

        return features, labels
    
    def __len__(self):
        """
        Get the total number of data samples in the dataset.

        Returns:
        --------
            - int: The total number of data samples.
        """

        return len(self.features)
    
dataset = AirbnbNightlyPriceRegressionDataset(data_nn_standardized)  # Use standardized data

# Define the sizes for train and test sets
total_size = len(dataset)
train_size = int(0.7 * total_size)  # 70% of the data for training
test_size = total_size - train_size   # Remaining for testing

# Split the dataset into train and test sets
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

train_data = next(iter(train_loader))
features, labels = train_data
print("Features:", len(features))
print("Labels:", len(labels))

def get_nn_config(config_file):
    """
    Load a neural network configuration from a YAML file.

    This function reads the configuration for a neural network model from a YAML file
    and returns it as a Python dictionary.

    Parameters:
    -----------
        - config_file (str): The path to the YAML configuration file.

    Returns:
    --------
        - dict: A dictionary containing the neural network configuration.
    """
    with open(config_file, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config

class NN(torch.nn.Module):
    """
    Neural Network Module for PyTorch.

    This module constructs a feedforward neural network for inference.

    Attributes:
    -----------
    model (nn.Sequential): The sequential model representing the neural network.

    Methods:
    --------
    __init__(self, config):
        - Initialize the neural network.

    forward(features):
        - Perform a forward pass through the neural network.
    """


    def __init__(self, config):
        """
        Initialize a feedforward neural network.

        Parameters:
        -----------
        - config (dict): A dictionary containing configuration parameters.
          Required keys:
            - "hidden_layer_width": Width of the hidden layers.
            - "depth": Depth of the neural network.
        """
        super(NN, self).__init__()  # Call the constructor of the parent class
        hidden_layer_width = config["hidden_layer_width"]
        depth = config["depth"]

        # Create a list of layers
        layers = []

        # Input layer
        layers.append(nn.Linear(11, hidden_layer_width))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_layer_width))  # BatchNorm after ReLU

        for _ in range(depth - 1):
            # Hidden layers
            layers.append(nn.Linear(hidden_layer_width, hidden_layer_width))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_layer_width))  # BatchNorm after ReLU

        # Output layer
        layers.append(nn.Linear(hidden_layer_width, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, features):
        """
        Perform a forward pass through the neural network.

        Parameters:
        -----------
        - features (torch.Tensor): Input tensor containing features.

        Returns:
        --------
        - torch.Tensor: Output tensor after passing through the network.
        """

        return self.model(features)
    
def calculate_accuracy(predictions, labels):
    """
    Calculate the accuracy of binary predictions.

    This function calculates the accuracy of binary predictions by rounding the
    predictions to the nearest integer (0 or 1) and comparing them to the actual labels.

    Parameters:
    -----------
        - predictions (torch.Tensor): Predicted values.
        - labels (torch.Tensor): Actual labels.

    Returns:
    --------
        - float: Accuracy as a percentage (0.0 to 1.0).
    """
    predictions = predictions.round()  # Round the predictions to the nearest integer (0 or 1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total

    return accuracy

def calculate_rmse(predictions,labels):
    """
    Calculate the Root Mean Squared Error (RMSE) between predictions and labels.

    This function computes the RMSE, a measure of the difference between predicted
    and actual values. Lower RMSE values indicate better model performance.

    Parameters:
    -----------
        - predictions (torch.Tensor): Predicted values.
        - labels (torch.Tensor): Actual labels.

    Returns:
    --------
        - float: The RMSE value.
    """
    mse = F.mse_loss(predictions,labels)
    rmse = math.sqrt(mse.item())

    return rmse

def calculate_r_squared(predictions, labels):
    """
    Calculate the coefficient of determination (R-squared) between predictions and labels.

    This function computes the R-squared value, which measures the proportion of the
    variance in the dependent variable that is predictable from the independent variable.
    
    Parameters:
    -----------
        - predictions (torch.Tensor): Predicted values.
        - labels (torch.Tensor): Actual labels.

    Returns:
    --------
        - float: The R-squared value, typically in the range [0, 1].
    """
    r2 = r2_score(labels.detach().numpy(), predictions.detach().numpy())

    return r2

def train(model, train_loader, test_loader, config, epochs):
    """
    Train a neural network model and evaluate its performance.

    This function trains a neural network model on a training dataset, evaluates its
    performance on a validation dataset, and logs various performance metrics using
    TensorBoardX.

    Parameters:
    -----------
        - model (torch.nn.Module): The neural network model to train and evaluate.
        - train_loader (DataLoader): DataLoader for the training dataset.
        - test_loader (DataLoader): DataLoader for the validation dataset.
        - config (dict): A dictionary containing configuration parameters.
        - epochs (int): The number of training epochs.

    Returns:
    --------
        - tuple: A tuple containing:
        - torch.nn.Module: The trained neural network model.
        - list of dict: Performance metrics for training and validation.
        - dict: Hyperparameters used for training.
    """
    learning_rate = config["learning_rate"]
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Create SummaryWriters for training and validation loss and accuracy
    writer = SummaryWriter()

    batch_count_train = 0
    batch_count_val = 0

    for epoch in range(epochs):
        model.train() # Set the model to training mode

        # Track start time for training
        start = time.time()

        for batch in train_loader:
            features, labels = batch
            prediction = model(features)
            labels = labels.view(-1, 1)  # Reshape labels to match the shape of prediction
            loss = F.mse_loss(prediction, labels) # computes the mean_squared_error

            # Calculate accuracy, RMSE and R^2
            accuracy = calculate_accuracy(prediction, labels)
            rmse = calculate_rmse(prediction,labels)
            r_squared = calculate_r_squared(prediction, labels)

            # Log training loss, accuracy, RMSE and R^2
            writer.add_scalar('Train Loss', loss.item(), batch_count_train)
            writer.add_scalar('Train Accuracy', accuracy, batch_count_train)
            writer.add_scalar('Train RMSE', rmse, batch_count_train)
            writer.add_scalar('Train R^2', r_squared, batch_count_train)

            # Zero the gradients, perform backward pass, and update the model parameters
            optimiser.zero_grad() 
            loss.backward() 
            optimiser.step() 

            batch_count_train += 1

            # Calculate training duration
            end = time.time()
            training_duration = end - start

        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        val_accuracy = 0.0
        val_rmse = 0.0
        val_r_squared = 0.0

        # Track start time for inference
        start_inference = time.time()

        for val_batch in test_loader:
            val_features, val_labels = val_batch
            val_labels = val_labels.view(-1, 1)  # Reshape val_labels to match val_prediction's shape
            val_prediction = model(val_features)
            val_loss += F.mse_loss(val_prediction, val_labels, reduction='sum').item()

            # Calculate accuracy, RMSE and R^2
            batch_accuracy = calculate_accuracy(val_prediction, val_labels)
            val_accuracy += batch_accuracy

            batch_rmse = calculate_rmse(val_prediction, val_labels)
            val_rmse += batch_rmse

            batch_r_squared = calculate_r_squared(val_prediction, val_labels)
            val_r_squared += batch_r_squared
        
            # Log validation loss, accuracy, RMSE, and R^2 score
            writer.add_scalar('Val Loss', val_loss, batch_count_val)
            writer.add_scalar('Val Accuracy', batch_accuracy, batch_count_val)
            writer.add_scalar('Val RMSE', batch_rmse, batch_count_val)
            writer.add_scalar('Val R^2', batch_r_squared, batch_count_val)

            batch_count_val += 1
            
            # Calculate inference latency for the entire validation set
            end_inference = time.time()
            inference_duration = end_inference - start_inference
            average_inference_latency = inference_duration / len(test_loader)

        val_loss /= len(test_dataset)
        val_accuracy /= len(test_loader)
        val_rmse /= len(test_loader)
        val_r_squared /= len(test_loader)    
       
        print("Epoch [{}/{}], "
                "Training Loss: {:.4f}, "
                "Training Accuracy: {:.4f}, "
                "Training RMSE: {:.4f}, "
                "Training R^2: {:.4f}, ".format(epoch+1, epochs, loss.item(), accuracy, rmse, r_squared))
            
        print("Validation Loss: {:.4f}, "
            "Validation Accuracy: {:.4f}, "
            "Validation RMSE: {:.4f}, "
            "Validation R^2: {:.4f}".format(val_loss, val_accuracy, val_rmse, val_r_squared))
        
    performance_metrics = [   
    {
            'Train Accuracy': accuracy,
            'Train RMSE': rmse,
            'Train r_squared': r_squared,
        },
    {
            'Val Accuracy': batch_accuracy,
            'Val RMSE': val_rmse,
            'Val r_squared': val_r_squared,
        }
    ] 

    hyperparameters = {
        'learning_rate': learning_rate,
        'hidden_layer_width': config["hidden_layer_width"],
        'depth': config["depth"]
    }    

    print("Training Duration: {:.2f} seconds, Average Inference Latency: {:.2f} seconds".format(round(training_duration, 4), round(average_inference_latency, 4)))
    
    return model, performance_metrics, hyperparameters

def save_model(model, hyperparams, metrics, folder):
    """
    Save the trained model, hyperparameters, and performance metrics to files.

    This function saves the trained neural network model, its associated hyperparameters,
    and the performance metrics evaluated on the validation data into separate files.

    Parameters:
    -----------
        - model: The best trained regression model.
        - hyperparams (dict): The best hyperparameters.
        - metrics (dict): Performance metrics on the validation data.
        - folder (str): The folder where the files will be saved.

    Returns:
    --------
        None
    """
    # Determine if the model is a PyTorch model
    is_torch_model = bool()

    try:
        model.state_dict()
        is_torch_model = True
    except:
        is_torch_model = False

    if is_torch_model == True:
        # Create a folder based on the current date and time
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        model_folder = os.path.join(folder, current_datetime)
        os.makedirs(model_folder, exist_ok=True)

        # Save the PyTorch model as 'model.pt'
        model_name = model.__class__.__name__
        model_path = os.path.join(model_folder, f"{model_name}_model.pt")
        torch.save(model.state_dict(), model_path)

        # Save hyperparameters to a JSON file
        hyperparams_path = os.path.join(model_folder, f"{model_name}_hyperparameters.json")
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f, indent=4)
    
        # Save performance metrics to a JSON file
        metrics_path = os.path.join(model_folder, f"{model_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
    else:
        # Save the model to a joblib file
        model_name = model.__class__.__name__
        model_path = os.path.join(folder, f"{model_name}_model.joblib")
        joblib.dump(model, model_path)
    
        # Save hyperparameters to a JSON file
        hyperparams_path = os.path.join(folder, f"{model_name}_hyperparameters.json")
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f, indent=4)
        
        # Save performance metrics to a JSON file
        metrics_path = os.path.join(folder, f"{model_name}_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

def generate_nn_configs():
    """
    Generate a list of neural network configuration dictionaries.

    This function generates a list of configuration dictionaries for neural network models
    by iterating over different hyperparameter values, including learning rates, hidden
    layer widths, and depths.

    Returns:
    --------
        - list: A list of dictionaries, where each dictionary represents a unique neural
        network configuration.
    """
    configs = []

    learning_rates = [0.001, 0.0001, 0.00001]
    hidden_layer_widths = [80, 100, 120]
    depths = [2, 3]

    for lr in learning_rates:
        for width in hidden_layer_widths:
            for depth in depths:
                config ={
                    "learning_rate": lr,
                    "hidden_layer_width": width,
                    "depth": depth
                }
                configs.append(config)

    return configs

def find_best_nn(train_loader, test_loader, epochs, n_splits=5):
    """
    Find the best neural network model configuration among multiple configurations using K-fold cross-validation.

    This function trains multiple neural network models with different configurations,
    evaluates their performance using K-fold cross-validation, and returns the best-performing model
    along with its metrics and hyperparameters.

    Parameters:
    -----------
        - train_loader (DataLoader): DataLoader for the training dataset.
        - test_loader (DataLoader): DataLoader for the test/validation dataset.
        - epochs (int): The number of training epochs.
        - n_splits (int): The number of K-fold splits (default is 5).

    Returns:
    --------
        - tuple: A tuple containing the best-performing model, its associated performance metrics,
        and the hyperparameters used to train the model.
    """
    best_val_rmse = 500
    best_model = None
    best_hyperparameters = None
    best_performance_metrics = None

    models_list = []
    hyperparameters_list = []
    performance_metrics_list = []

    # Create K-fold cross-validation iterator
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    for config in generate_nn_configs():
        print(config)
        val_rmse_sum = 0

        for train_index, test_index in kf.split(train_loader.dataset):
            train_dataset, test_dataset = torch.utils.data.Subset(train_loader.dataset, train_index), torch.utils.data.Subset(train_loader.dataset, test_index)

            model = NN(config)
            model, performance_metrics, hyperparameters = train(model, train_loader, test_loader, config, epochs)

            val_dict = performance_metrics[1]
            val_rmse = val_dict['Val RMSE']
            val_rmse_sum += val_rmse

        # Calculate the average RMSE over K folds
        avg_val_rmse = val_rmse_sum / n_splits

        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            best_model = model
            best_performance_metrics = performance_metrics
            best_hyperparameters = hyperparameters

        models_list.append(best_model)
        performance_metrics_list.append(best_performance_metrics)
        hyperparameters_list.append(best_hyperparameters)

    save_model(best_model, best_hyperparameters, best_performance_metrics, 'models/neural_networks/regression/kfold')

    return best_model, best_performance_metrics, best_hyperparameters

if __name__ == "__main__":

    config = get_nn_config("nn_config.yaml")
    model = NN(config)
    train(model, train_loader, test_loader, config, epochs=150)
    best_model, best_performance_metrics, best_hyperparameters = find_best_nn(train_loader, test_loader, epochs=150)

    print("Best model:", best_model)
    print("Best performance metrics:", best_performance_metrics)
    print("Best hyperparameters:", best_hyperparameters)