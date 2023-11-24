from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, precision_score , recall_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from tabular_data import load_airbnb
import joblib
import json
import numpy as np
import os 
import datetime

# Define the path to the data file
file_path = '/Users/andreasyianni/Desktop/Educational Training /Data Scientist Training/AiCore Training/Projects/Modelling-Airbnbs-Property-Listing-Dataset/airbnb-property-listings/tabular_data/clean_tabular_data.csv'

# Load the Airbnb data and split it into training and valuation sets 
data = load_airbnb(file_path, 'Category')
X, y = data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LogisticRegression model and train it
model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)

# Make predictions using the trained model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_val)

# Calculate performance metrics for the training set
train_f1_score = f1_score(y_train, y_train_pred, average='macro')
train_precision = precision_score(y_train, y_train_pred, average='macro')
train_recall = recall_score(y_train, y_train_pred, average='macro')
train_accuracy = accuracy_score(y_train, y_train_pred)

# Calculate performance metrics for the test set
test_f1_score = f1_score(y_val, y_test_pred, average='macro')
test_precision = precision_score(y_val, y_test_pred, average='macro')
test_recall = recall_score(y_val, y_test_pred, average='macro')
test_accuracy = accuracy_score(y_val, y_test_pred)

# Print the computed metrics
print("Training Set Metrics:")
print(f"F1 Score: {train_f1_score}")
print(f"Precision: {train_precision}")
print(f"Recall: {train_recall}")
print(f"Accuracy: {train_accuracy}")

print("\nTest Set Metrics:")
print(f"F1 Score: {test_f1_score}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"Accuracy: {test_accuracy}")

def tune_classification_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, hyperparameters):
    """
    Tune hyperparameters of a classification model using grid search and evaluate its performance on validation data.

    This function performs hyperparameter tuning for a classification model using GridSearchCV
    from scikit-learn. It fits the model on the training data, searches over the provided
    hyperparameter grid using cross-validation, and selects the best model based on the
    specified scoring metric.

    Parameters:
    -----------
        - model_class (class): The classification model class to be tuned.
        - X_train (array-like): Training data features.
        - y_train (array-like): Training data labels.
        - X_val (array-like): Validation data features.
        - y_val (array-like): Validation data labels.
        - hyperparameters (dict): Hyperparameter grid for grid search.

    Returns:
    --------
        - model: The tuned classification model.
        - hyperparams (dict): The hyperparameters.
        - performance_metrics (dict): Performance metrics on the validation data.
    """
    model = model_class
    model.fit(X_train, y_train)
    model.predict(X_val)

    grid = GridSearchCV(model_class, param_grid=hyperparameters, refit = True, verbose = 3,n_jobs=-1)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    hyperparams = grid.best_params_

    val_predictions = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    performance_metrics = {"validation_accuracy": val_accuracy}

    return model, hyperparams, performance_metrics

def save_model(model, hyperparams, metrics, folder):
    """
    Save the trained model, hyperparameters, and performance metrics to files.

    This function saves the trained classification model, its associated hyperparameters,
    and the performance metrics evaluated on the validation data into separate files.

    Parameters:
    -----------
        - model: The best trained regression model.
        - hyperparams (dict): The best hyperparameters.
        - metrics (dict): Performance metrics on the validation data.
        - folder (str): The folder where the files will be saved.

    Returns:
    --------
       - None
    """
    # Get the model class name
    model_name = model.__class__.__name__

    # Save the model to a joblib file
    model_path = os.path.join(folder, f"{model_name}_model.joblib")
    joblib.dump(model, model_path)

    # Save hyperparameters to a JSON file
    hyperparams_path = os.path.join(folder,  f"{model_name}_hyperparameters.json")
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparams, f, indent=4)
    
    # Save performance metrics to a JSON file
    metrics_path = os.path.join(folder, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

def evaluate_all_models(X_train, y_train, X_val, y_val, hyperparameters, models):
    """
    Perform hyperparameter tuning and evaluation for multiple classification models.

    This function iterates over a list of classification models and their corresponding
    hyperparameters. For each model, it tunes the hyperparameters, evaluates its
    performance on the validation dataset, and saves the best model, hyperparameters,
    and performance metrics for classification models.

    Parameters:
    -----------
        - X_train (array-like): Training data features.
        - y_train (array-like): Training data labels.
        - X_val (array-like): Validation data features.
        - y_val (array-like): Validation data labels.
        - hyperparameters (list of dict): List of hyperparameters for different classification models.
        - models (list of classifier models): List of classification model instances.

    Returns:
    --------
        None
    """
    # Iterate over the models and hyperparameters and perform tuning and evaluation for classification models
    for model_config, model_class in zip(hyperparameters, models):
        model_name = model_config['model_name']
        hyperparameters_list = model_config['hyperparameters']

        best_model, best_hyperparams, performance_metrics = tune_classification_model_hyperparameters(
            model_class=model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hyperparameters=hyperparameters_list)

        print(f"Model: {model_name}")
        print("Best Estimator:", best_model)
        print("Best Hyperparameters:", best_hyperparams)
        print("Performance_metrics:", performance_metrics)

        # Save the model, hyperparameters, and metrics
        save_model(best_model, best_hyperparams, performance_metrics, folder = "models/classification/logistic_regression")

def find_best_model(X_train, y_train, X_val, y_val, hyperparameters, models):
    """
    Find the best classification models among a list of models by evaluating their performance on
    validation data.

    This function iterates through a list of classification models along with their corresponding hyperparameters,
    tunes each model, evaluates its performance on the validation dataset, and returns the best classification model,
    its best hyperparameters, and the performance metrics for the best model.

    Parameters:
    -----------
        - X_train (array-like): Training data features.
        - y_train (array-like): Training data labels.
        - X_val (array-like): Validation data features.
        - y_val (array-like): Validation data labels.
        - hyperparameters (list of dict): List of hyperparameters for different classification models.
        - models (list of classifier models): List of classification model instances.

    Returns:
    --------
        - best_model: The best trained classification model.
        - best_hyperparams (dict): The best hyperparameters for classification.
        - best_performance_metrics (dict): Performance metrics for the best classification model.
    """
    best_model = None
    best_hyperparams = None
    best_performance_metrics = {"validation_accuracy": 0}

    for model_config, model_class in zip(hyperparameters, models):
        model_name = model_config['model_name']
        hyperparameters_list = model_config['hyperparameters']

        best_model, best_hyperparams, performance_metrics = tune_classification_model_hyperparameters(
            model_class=model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hyperparameters=hyperparameters_list)

        print(f"Model: {model_name}")
        print("Best Estimator:", best_model)
        print("Best Hyperparameters:", best_hyperparams)
        print("Performance_metrics:", performance_metrics)

        if performance_metrics["validation_accuracy"] > best_performance_metrics["validation_accuracy"]:
            best_model = best_model
            best_hyperparams = best_hyperparams
            best_performance_metrics = performance_metrics

    return (best_model, best_hyperparams, best_performance_metrics)

if __name__ == "__main__":

    # Define a dictionary of hyperparameters for different classification models
    hyperparameters = [
    {
        'model_name': 'GradientBoosting',
        'hyperparameters': {
            'loss': ['log_loss'],
            'n_estimators': [10, 100, 1000],
            'criterion': ['squared_error', 'friedman_mse'],
            'learning_rate': [0.1, 0.001, 0.0001]
        }
    },
    {
        'model_name': 'DecisionTree',
        'hyperparameters': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
    },
    {
        'model_name': 'RandomForest',
        'hyperparameters': {
            'n_estimators': [10, 100, 150],
            'max_depth': [None, 5, 10, 20],
            'max_features': ['sqrt', 'log2'],
            'min_samples_split': [2, 5, 10]
        }
    },
    {
        'model_name': 'LogisticRegression',
        'hyperparameters': {
            'penalty': [ 'l2'],
            'C': [0.1, 1],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [5000, 10000, 15000]

        }
    },
    {
        'model_name': 'SGDClassifier',
        'hyperparameters': {
            'loss': ['hinge', 'log_loss'],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'max_iter': [1000, 5000],
            'alpha': [0.0001, 0.001, 0.01]    
            }
    }
]
    # Define a list of all the classification models
    models = [GradientBoostingClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression(), SGDClassifier()]
        
    evaluate_all_models(X_train, y_train, X_val, y_val, hyperparameters, models)

    (best_model, best_hyperparams,best_performance_metrics) = find_best_model( X_train, y_train, X_val, y_val, 
                                                                              hyperparameters, models)

    # Print the best model, hyperparameters, and performance metrics for classification models
    print("\nClassification Model\n")
    print("Best Model:", best_model)
    print("Best Hyperparameters:", best_hyperparams)
    print("Best Performance Metrics:", best_performance_metrics)


