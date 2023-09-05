from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import SGDRegressor 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from tabular_data import load_airbnb
import itertools
import joblib
import json
import numpy as np
import os 
import pandas as pd

# Define the path to the data file
file_path = '/Users/andreasyianni/Desktop/Educational Training /Data Scientist Training/AiCore Training/Projects/Modelling-Airbnbs-Property-Listing-Dataset/airbnb-property-listings/tabular_data/clean_tabular_data.csv'

# Load the Airbnb data and split it into features (X) and target (y)
data = load_airbnb(file_path, "Price_Night")
X, y = data

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a SGDRegressor model and train it# Create a SGDRegressor model and train it
model = SGDRegressor(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions using the trained model
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE) and R-squared (R2) for evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, hyperparameters):
    """
    Tune hyperparameters of a regression model using custom grid search.

    Parameters:
        - model_class (class): The scikit-learn regressor class to be tuned.
        - X_train (array-like): Training data features.
        - y_train (array-like): Training data labels.
        - X_val (array-like): Validation data features.
        - y_val (array-like): Validation data labels.
        - hyperparameters (dict): Hyperparameter grid for grid search.

    Returns:
        - best_model: The best trained model.
        - best_hyperparams (dict): The best hyperparameters.
        - performance_metrics (dict): Performance metrics on the validation data.
    """
    def grid_search(hyperparameters):
        keys, values = zip(*hyperparameters.items())
        yield from (dict(zip(keys, v)) for v in itertools.product(*values))

    best_hyperparams, best_loss = None, np.inf

    for hyperparams in grid_search(hyperparameters):
        model = model_class(**hyperparams)
        model.fit(X_train, y_train)

        y_validation_pred = model.predict(X_val)
        validation_loss = mean_squared_error(y_val, y_validation_pred)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_hyperparams = hyperparams

    best_model = model_class(**best_hyperparams)
    best_model.fit(X_train, y_train)
    
    y_test_pred = best_model.predict(X_val)
    test_rmse = np.sqrt(mean_squared_error(y_val, y_test_pred))
    
    performance_metrics = {"validation_RMSE": test_rmse}
    
    return best_model, best_hyperparams, performance_metrics

def tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, hyperparameters):
    """
    Tune hyperparameters of a regression model using GridSearchCV.

    Parameters:
        - model_class (class): The scikit-learn regressor class to be tuned.
        - X_train (array-like): Training data features.
        - y_train (array-like): Training data labels.
        - X_val (array-like): Validation data features.
        - y_val (array-like): Validation data labels.
        - hyperparameters (dict): Hyperparameter grid for grid search.

    Returns:
        - best_estimator: The best trained model.
        - best_params (dict): The best hyperparameters.
        - performance_metrics (dict): Performance metrics on the validation data.
    """
    model = model_class
    model.fit(X_train, y_train)
    model.predict(X_val)

    grid = GridSearchCV(model_class, param_grid=hyperparameters, refit = True, verbose = 3,n_jobs=-1) 
    grid.fit(X_train, y_train)

    best_estimator = grid.best_estimator_
    best_params = grid.best_params_

    grid_predictions = grid.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_val, grid_predictions))

    performance_metrics = {"validation_RMSE": rmse}

    return best_estimator, best_params, performance_metrics


def save_model(model, hyperparams, metrics, model_name, folder):
    """
    Save the best regression model, hyperparameters, and performance metrics to files.

    Parameters:
        - model: The best trained regression model.
        - hyperparams (dict): The best hyperparameters.
        - metrics (dict): Performance metrics on the validation data.
        - model_name (str): The name of the model.
        - folder (str): The folder where the files will be saved.
    """

    # Save the model to a joblib file
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

def evaluate_all_models(X_train, y_train, X_val, y_val, hyperparameters_list, models_list):
    """
    This function performs hyperparameter tuning and evaluation for multiple regression models using the provided
    training and validation datasets. It iterates over a list of regression models and their corresponding
    hyperparameters, tunes each model, evaluates its performance on the validation dataset, and saves the best model,
    hyperparameters, and performance metrics.

    Parameters:
        - X_train (array-like): Training data features.
        - y_train (array-like): Training data labels.
        - X_val (array-like): Validation data features.
        - y_val (array-like): Validation data labels.
        - hyperparameters_list (list of dict): List of hyperparameters for different regression models.
        - models_list (list of regressor models): List of regression model instances.

    Returns:
        None
    """
# Iterate over the models and hyperparameters and perform tuning and evaluation
    for model_config, model_class in zip(hyperparameters_list, models_list):
        model_name = model_config['model_name']
        hyperparameters_list = model_config['hyperparameters']

        best_estimator, best_params, performance_metrics = tune_regression_model_hyperparameters(
            model_class=model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hyperparameters=hyperparameters_list)

        print(f"Model: {model_name}")
        print("Best Estimator:", best_estimator)
        print("Best Hyperparameters:", best_params)
        print("Performance_metrics:", performance_metrics)

        # Save the model, hyperparameters, and metrics
        save_model(best_estimator, best_params, performance_metrics, model_name, folder="models/regression")

def find_best_model(X_train, y_train, X_val, y_val, hyperparameters_list, models_list):
    """
    Find the best regression model among a list of models by evaluating their performance on validation data.

    Parameters:
        - X_train (array-like): Training data features.
        - y_train (array-like): Training data labels.
        - X_val (array-like): Validation data features.
        - y_val (array-like): Validation data labels.
        - hyperparameters_list (list of dict): List of hyperparameters for different regression models.
        - models_list (list of regressor models): List of regression model instances.

    Returns:
        - best_model: The best trained model.
        - best_hyperparams (dict): The best hyperparameters.
        - performance_metrics (dict): Performance metrics on the validation data.
    """
    best_model = None
    best_hyperparams = None
    best_performance_metrics = {"validation_RMSE": float("inf")}

    for model_config, model_class in zip(hyperparameters_list, models_list):
        model_name = model_config['model_name']
        hyperparameters_list = model_config['hyperparameters']

        best_estimator, best_params, performance_metrics = tune_regression_model_hyperparameters(
            model_class=model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hyperparameters=hyperparameters_list)

        print(f"Model: {model_name}")
        print("Best Estimator:", best_estimator)
        print("Best Hyperparameters:", best_params)
        print("Performance_metrics:", performance_metrics)

        if performance_metrics["validation_RMSE"] < best_performance_metrics["validation_RMSE"]:
            best_model = best_estimator
            best_hyperparams = best_params
            best_performance_metrics = performance_metrics

    return best_model, best_hyperparams, best_performance_metrics


if __name__ == "__main__":
    data = load_airbnb(file_path, "Price_Night")
    X, y = data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    # Define hyperparameter grid
    grid = {
        "max_iter": [1000, 2000],
        "alpha": [0.0001, 0.001],
        "learning_rate": ['constant', 'invscaling', 'adaptive']
    }

    best_model, best_hyperparams, performance_metrics = custom_tune_regression_model_hyperparameters(
        model_class=SGDRegressor,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        hyperparameters=grid
    )
    print("Best Model:", best_model)
    print("Best Hyperparameters:", best_hyperparams)
    print("Performance Metrics:", performance_metrics)

    # Define a dictionary of hyperparameters for different regression models
    hyperparameters_list = [
        {
            'model_name': 'GradientBoosting',
            'hyperparameters': {
                'loss': ['squared_error', 'huber'],
                'n_estimators': [10, 100, 1000],
                'criterion': ['squared_error', 'friedman_mse'],
                'learning_rate': [0.1, 0.001, 0.0001]
            }
        },
        {
            'model_name': 'SVM',
            'hyperparameters': {
                'C': [1],
                'gamma': ['auto'],
                'kernel': ['linear']
            }
        },
        {
            'model_name': 'DecisionTree',
            'hyperparameters': {
                'criterion': ['mse', 'friedman_mse'],
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
            'model_name': 'SGD',
            'hyperparameters': {
                'loss': ['squared_error'],
                'penalty': ['l2', 'elasticnet'],
                'learning_rate': ['invscaling', 'optimal'],
                'max_iter': [1000, 2000]
            }
        }
    ]

    # Define a list of all the regression models
    models_list = [GradientBoostingRegressor(), SVR(), DecisionTreeRegressor(), RandomForestRegressor(), SGDRegressor()]

    evaluate_all_models(X_train, y_train, X_val, y_val, hyperparameters_list, models_list)

    best_model, best_hyperparams, best_performance_metrics = find_best_model(X_train, y_train, X_val, y_val, hyperparameters_list, models_list)
    
    # Print the best model, hyperparameters, and performance metrics
    print("Best Model:", best_model)
    print("Best Hyperparameters:", best_hyperparams)
    print("Best Performance Metrics:", best_performance_metrics)