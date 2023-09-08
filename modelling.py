from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import SGDRegressor, LogisticRegression, SGDClassifier
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_score , recall_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from tabular_data import load_airbnb
import itertools
import joblib
import json
import numpy as np
import os 

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

def tune_classification_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, hyperparameters):
    
    model = model_class
    model.fit(X_train, y_train)
    model.predict(X_val)

    grid = GridSearchCV(model_class, param_grid=hyperparameters, refit = True, verbose = 3,n_jobs=-1)
    grid.fit(X_train, y_train)

    best_estimator = grid.best_estimator_
    best_params = grid.best_params_

    val_predictions = best_estimator.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)

    performance_metrics = {"validation_accuracy": val_accuracy}

    return best_estimator, best_params, performance_metrics

def save_model(model, hyperparams, metrics, folder):
    """
    Save the best regression model, hyperparameters, and performance metrics to files.

    Parameters:
        - model: The best trained regression model.
        - hyperparams (dict): The best hyperparameters.
        - metrics (dict): Performance metrics on the validation data.
        - model_name (str): The name of the model.
        - folder (str): The folder where the files will be saved.
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

def evaluate_all_models(X_train_reg, y_train_reg, X_val_reg, y_val_reg, regression_hyperparameters,
                        X_train_cls, y_train_cls, X_val_cls, y_val_cls, classification_hyperparameters,
                        regression_models, classification_models, task_folder):
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
# Iterate over the models and hyperparameters and perform tuning and evaluation for regression models
    for model_config, model_class in zip(regression_hyperparameters, regression_models):
        model_name = model_config['model_name']
        hyperparameters_list = model_config['hyperparameters']

        best_estimator, best_params, performance_metrics = tune_regression_model_hyperparameters(
            model_class=model_class,
            X_train=X_train_reg,
            y_train=y_train_reg,
            X_val=X_val_reg,
            y_val=y_val_reg,
            hyperparameters=hyperparameters_list)

        print(f"Model: {model_name}")
        print("Best Estimator:", best_estimator)
        print("Best Hyperparameters:", best_params)
        print("Performance_metrics:", performance_metrics)

        # Save the model, hyperparameters, and metrics
        task_folder = "models/regression"
        save_model(best_estimator, best_params, performance_metrics, task_folder)
    
    # Iterate over the models and hyperparameters and perform tuning and evaluation for regression models
    for model_config, model_class in zip(classification_hyperparameters, classification_models):
        model_name = model_config['model_name']
        hyperparameters_list = model_config['hyperparameters']

        best_estimator, best_params, performance_metrics = tune_classification_model_hyperparameters(
            model_class=model_class,
            X_train=X_train_cls,
            y_train=y_train_cls,
            X_val=X_val_cls,
            y_val=y_val_cls,
            hyperparameters=hyperparameters_list)

        print(f"Model: {model_name}")
        print("Best Estimator:", best_estimator)
        print("Best Hyperparameters:", best_params)
        print("Performance_metrics:", performance_metrics)

        # Save the model, hyperparameters, and metrics
        task_folder = "models/classification"
        save_model(best_estimator, best_params, performance_metrics, task_folder)

def find_best_model(X_train_reg, y_train_reg, X_val_reg, y_val_reg, regression_hyperparameters,
                    X_train_cls, y_train_cls, X_val_cls, y_val_cls, classification_hyperparameters,
                    regression_models, classification_models):
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
    best_model_reg = None
    best_hyperparams_reg = None
    best_performance_metrics_reg = {"validation_RMSE": float("inf")}

    for model_config, model_class in zip(regression_hyperparameters, regression_models):
        model_name = model_config['model_name']
        hyperparameters_list = model_config['hyperparameters']

        best_estimator_reg, best_hyperparams_reg, performance_metrics_reg = tune_regression_model_hyperparameters(
            model_class=model_class,
            X_train=X_train_reg,
            y_train=y_train_reg,
            X_val=X_val_reg,
            y_val=y_val_reg,
            hyperparameters=hyperparameters_list)

        print(f"Model: {model_name}")
        print("Best Estimator:", best_estimator_reg)
        print("Best Hyperparameters:", best_hyperparams_reg)
        print("Performance_metrics:", performance_metrics_reg)

        if performance_metrics_reg["validation_RMSE"] < best_performance_metrics_reg["validation_RMSE"]:
            best_model_reg = best_estimator_reg
            best_hyperparams_reg = best_hyperparams_reg
            best_performance_metrics_reg = performance_metrics_reg

        # best_metrics_reg = best_model_reg, best_hyperparams, best_performance_metrics_reg
    
    best_model_cls = None
    best_hyperparams_cls = None
    best_performance_metrics_cls = {"validation_accuracy": 0}

    for model_config, model_class in zip(classification_hyperparameters, classification_models):
        model_name = model_config['model_name']
        hyperparameters_list = model_config['hyperparameters']

        best_estimator_cls, best_hyperparams_cls, performance_metrics_cls = tune_classification_model_hyperparameters(
            model_class=model_class,
            X_train=X_train_cls,
            y_train=y_train_cls,
            X_val=X_val_cls,
            y_val=y_val_cls,
            hyperparameters=hyperparameters_list)

        print(f"Model: {model_name}")
        print("Best Estimator:", best_estimator_cls)
        print("Best Hyperparameters:", best_hyperparams_cls)
        print("Performance_metrics:", performance_metrics_cls)

        if performance_metrics_cls["validation_accuracy"] > best_performance_metrics_cls["validation_accuracy"]:
            best_model_cls = best_estimator_cls
            best_hyperparams_cls = best_hyperparams_cls
            best_performance_metrics_cls = performance_metrics_cls

        # best_metrics_cls = best_model_cls, best_hyperparams, best_performance_metrics_cls

    return (best_model_reg, best_hyperparams_reg, best_performance_metrics_reg,
            best_model_cls, best_hyperparams_cls, best_performance_metrics_cls)


if __name__ == "__main__":

    data_price_night = load_airbnb(file_path, "Price_Night")
    X_reg, y_reg = data_price_night
    X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

    # Define hyperparameter grid
    grid = {
        "max_iter": [1000, 2000],
        "alpha": [0.0001, 0.001],
        "learning_rate": ['constant', 'invscaling', 'adaptive']
    }

    best_model, best_hyperparams, performance_metrics = custom_tune_regression_model_hyperparameters(
        model_class=SGDRegressor,
        X_train=X_train_reg,
        y_train=y_train_reg,
        X_val=X_val_reg,
        y_val=y_val_reg,
        hyperparameters=grid
    )
    print("Best Model:", best_model)
    print("Best Hyperparameters:", best_hyperparams)
    print("Performance Metrics:", performance_metrics)

    # Define a dictionary of hyperparameters for different regression models
    regression_hyperparameters = [
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

    

   

    # Load the Airbnb data and split it into features (X) and target (y)
    data_category = load_airbnb(file_path, 'Category')
    X_cls, y_cls = data_category

    # Split the data into training and testing sets
    X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

    hyperparameters_sgd = {
                      'loss': ['squared_error'],
                      'penalty': ['elasticnet'],
                      'max_iter': [1000, 2000],
                      'alpha': [0.0001, 0.001]
                       }

    classification_hyperparameters = [
    {
        'model_name': 'GradientBoosting',
        'hyperparameters': {
            'loss': ['deviance'],
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
    # {
    #     'model_name': 'SGDClassifier',
    #     'hyperparameters': {
    #         'loss': ['log', 'modified_huber'],
    #         'penalty': ['elasticnet'],
    #         'max_iter': [8000, 10000],
    #         'alpha': [0.0001, 0.001, 0.01],      
    #         }
    # }
]

# Define a list of all the regression models
regression_models = [GradientBoostingRegressor(), SVR(), DecisionTreeRegressor(), RandomForestRegressor(), SGDRegressor()]

# Define a list of all the classification models
classification_models = [GradientBoostingClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), SGDRegressor()]
    
# evaluate_all_models(X_train_cls, y_train_cls, X_val_cls, y_val_cls, classification_hyperparameters, regression_models, classification_models, task_folder)

# best_model, best_hyperparams, best_performance_metrics = find_best_model(X_train, y_train, X_val, y_val, hyperparameters_regression, regression_models_list)

(best_model_reg, best_hyperparams_reg, best_performance_metrics_reg, best_model_cls, best_hyperparams_cls, 
best_performance_metrics_cls) = find_best_model(X_train_reg, y_train_reg, X_val_reg, 
                                                    y_val_reg, regression_hyperparameters, 
                                                    X_train_cls, y_train_cls, X_val_cls, 
                                                    y_val_cls, classification_hyperparameters, 
                                                    regression_models, classification_models)

# Print the best model, hyperparameters, and performance metrics for regression models
print("\nRegression Model\n")
print("Best Model:", best_model_reg)
print("Best Hyperparameters:", best_hyperparams_reg)
print("Best Performance Metrics:", best_performance_metrics_reg)

# Print the best model, hyperparameters, and performance metrics for classification models
print("\nClassification Model\n")
print("Best Model:", best_model_cls)
print("Best Hyperparameters:", best_hyperparams_cls)
print("Best Performance Metrics:", best_performance_metrics_cls)

