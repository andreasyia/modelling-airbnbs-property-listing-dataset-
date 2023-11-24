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

# Define the path to the data file
file_path = '/Users/andreasyianni/Desktop/Educational Training /Data Scientist Training/AiCore Training/Projects/Modelling-Airbnbs-Property-Listing-Dataset/airbnb-property-listings/tabular_data/clean_tabular_data.csv'

# Load the Airbnb data and split it into training and valuation sets 
data = load_airbnb(file_path, "Price_Night")
X, y = data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a SGDRegressor model and train it
model = SGDRegressor(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions using the trained model
y_pred = model.predict(X_val)

# Calculate Mean Squared Error (MSE) and R-squared (R2) for evaluation
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

def custom_tune_regression_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, hyperparameters):
    """
    Tune hyperparameters of a regression model using a custom grid search approach.

    This function tunes the hyperparameters of a regression model using a custom grid search
    method. It iterates over the provided hyperparameter grid and trains the model with each
    combination. The best model is selected based on the lowest validation root mean squared
    error (RMSE).

    Parameters:
    -----------
        - model_class (class): The scikit-learn regressor class to be tuned.
        - X_train (array-like): Training data features.
        - y_train (array-like): Training data labels.
        - X_val (array-like): Validation data features.
        - y_val (array-like): Validation data labels.
        - hyperparameters (dict): Hyperparameter grid for grid search.

    Returns:
    --------
        - best_model: The trained model.
        - best_hyperparams (dict): The hyperparameters.
        - performance_metrics (dict): Performance metrics on the validation data.
    """
    def grid_search(hyperparameters):
        """
        Generate a grid search space for hyperparameters.

        This function takes a dictionary of hyperparameters and generates a grid search space
        by computing all possible combinations of hyperparameter values.

        Parameters:
        -----------
        hyperparameters (dict): A dictionary where keys are hyperparameter names and values are lists of
                                values for each hyperparameter.
            
        Yields:
        -------
        dict: Yields dictionaries representing each unique combination of hyperparameters.
        """
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

    This function performs hyperparameter tuning for a regression model using GridSearchCV
    from scikit-learn. It fits the model on the training data, searches over the provided
    hyperparameter grid using cross-validation, and selects the best model based on the
    specified scoring metric.

    Parameters:
    -----------
        - model_class (class): The scikit-learn regressor class to be tuned.
        - X_train (array-like): Training data features.
        - y_train (array-like): Training data labels.
        - X_val (array-like): Validation data features.
        - y_val (array-like): Validation data labels.
        - hyperparameters (dict): Hyperparameter grid for grid search.

    Returns:
    --------
        - model: The trained model.
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

    grid_predictions = grid.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, grid_predictions))

    performance_metrics = {"validation_RMSE": rmse}

    return model, hyperparams, performance_metrics

def save_model(model, hyperparams, metrics, folder):
    """
    Save the trained model, hyperparameters, and performance metrics to files.

    This function saves the trained regreession model, its associated hyperparameters,
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
    # Get the model class name
    model_name = model.__class__.__name__

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

def evaluate_all_models(X_train, y_train, X_val, y_val, hyperparameters, models):
    """
    Perform hyperparameter tuning and evaluation for multiple regression models.

    This function performs hyperparameter tuning and evaluation for multiple regression models
    using the provided training and validation datasets. It iterates over a list of regression models,
    their corresponding hyperparameters, tunes each model, evaluates its performance on the validation dataset,
    and saves the best model, hyperparameters, and performance metrics for regression models.

    Parameters:
    -----------
        - X_train (array-like): Training data features.
        - y_train (array-like): Training data labels.
        - X_val (array-like): Validation data features.
        - y_val (array-like): Validation data labels.
        - hyperparameters (list of dict): List of hyperparameters for different regression models.
        - models (list of regressor models): List of regression model instances.

    Returns:
    --------
        - None
    """
# Iterate over the models and hyperparameters and perform tuning and evaluation for regression models
    for model_config, model_class in zip(hyperparameters, models):
        model_name = model_config['model_name']
        hyperparameters_list = model_config['hyperparameters']

        best_model, best_params, performance_metrics = tune_regression_model_hyperparameters(
            model_class=model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hyperparameters=hyperparameters_list)

        print(f"Model: {model_name}")
        print("Best Estimator:", best_model)
        print("Best Hyperparameters:", best_params)
        print("Performance_metrics:", performance_metrics)

        # Save the model, hyperparameters, and metrics
        save_model(best_model, best_params, performance_metrics, folder = "models/regression/linear_regression")

def find_best_model(X_train, y_train, X_val, y_val, hyperparameters, models):
    """
    Find the best regression model among a list of models by evaluating their performance on
    validation data.

    This function iterates through a list of regression models along with their corresponding hyperparameters,
    tunes each model, evaluates its performance on the validation dataset, and returns the best regression model,
    its best hyperparameters, and the performance metrics for the best model.

    Parameters:
    -----------
        - X_train (array-like): Training data features.
        - y_train (array-like): Training data labels.
        - X_val (array-like): Validation data features.
        - y_val (array-like): Validation data labels.
        - hyperparameters (list of dict): List of hyperparameters for different regression models.
        - models (list of regressor models): List of regression model instances.
        
    Returns:
    --------
        - best_model: The best trained regression model.
        - best_hyperparams (dict): The best hyperparameters for regression.
        - best_performance_metrics (dict): Performance metrics for the best regression model.
    """
    best_model = None
    best_hyperparams = None
    best_performance_metrics = {"validation_RMSE": float("inf")}

    for model_config, model_class in zip(hyperparameters, models):
        model_name = model_config['model_name']
        hyperparameters_list = model_config['hyperparameters']

        best_estimator, best_hyperparams, performance_metrics = tune_regression_model_hyperparameters(
            model_class=model_class,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            hyperparameters=hyperparameters_list)

        print(f"Model: {model_name}")
        print("Best Estimator:", best_estimator)
        print("Best Hyperparameters:", best_hyperparams)
        print("Performance_metrics:", performance_metrics)

        if performance_metrics["validation_RMSE"] < best_performance_metrics["validation_RMSE"]:
            best_model = best_estimator
            best_hyperparams = best_hyperparams
            best_performance_metrics = performance_metrics    
    
    return (best_model, best_hyperparams, best_performance_metrics)

if __name__ == "__main__":

    # Define hyperparameter for 'grid_search'
    grid = {
        "max_iter": [1000, 2000],
        "alpha": [0.0001, 0.001],
        "learning_rate": ['constant', 'invscaling', 'adaptive']
    }

    # Define hyperparameter for SGDRegressor
    hyperparameters_sgd = {
                      'loss': ['squared_error'],
                      'penalty': ['elasticnet'],
                      'max_iter': [1000, 2000],
                      'alpha': [0.0001, 0.001]
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
    hyperparameters = [
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
    models = [GradientBoostingRegressor(), SVR(), DecisionTreeRegressor(), RandomForestRegressor(), SGDRegressor()]
    
    evaluate_all_models(X_train, y_train, X_val, y_val, hyperparameters, models)

    (best_model, best_hyperparams, best_performance_metrics) = find_best_model(X_train, y_train, X_val, y_val, 
                                                                                hyperparameters, models)

    # Print the best model, hyperparameters, and performance metrics for regression models
    print("\nRegression Model\n")
    print("Best Model:", best_model)
    print("Best Hyperparameters:", best_hyperparams)
    print("Best Performance Metrics:", best_performance_metrics)


