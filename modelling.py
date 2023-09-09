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

# Load the Airbnb data and split it into training and valuation sets 

#Price_night
data_price_night = load_airbnb(file_path, "Price_Night")
X_reg, y_reg = data_price_night
X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Category
data_category = load_airbnb(file_path, 'Category')
X_cls, y_cls = data_category
X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# Create a SGDRegressor model and train it
model = SGDRegressor(max_iter=1000, random_state=42)
model.fit(X_train_reg, y_train_reg)

# Make predictions using the trained model
y_pred = model.predict(X_val_reg)

# Calculate Mean Squared Error (MSE) and R-squared (R2) for evaluation
mse = mean_squared_error(y_val_reg, y_pred)
r2 = r2_score(y_val_reg, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Create a LogisticRegression model and train it
model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_train_cls, y_train_cls)

# Make predictions using the trained model
y_train_pred = model.predict(X_train_cls)
y_test_pred = model.predict(X_val_cls)

# Calculate performance metrics for the training set
train_f1_score = f1_score(y_train_cls, y_train_pred, average='macro')
train_precision = precision_score(y_train_cls, y_train_pred, average='macro')
train_recall = recall_score(y_train_cls, y_train_pred, average='macro')
train_accuracy = accuracy_score(y_train_cls, y_train_pred)

# Calculate performance metrics for the test set
test_f1_score = f1_score(y_val_cls, y_test_pred, average='macro')
test_precision = precision_score(y_val_cls, y_test_pred, average='macro')
test_recall = recall_score(y_val_cls, y_test_pred, average='macro')
test_accuracy = accuracy_score(y_val_cls, y_test_pred)

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

    grid_predictions = grid.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, grid_predictions))

    performance_metrics = {"validation_RMSE": rmse}

    return best_estimator, best_params, performance_metrics

def tune_classification_model_hyperparameters(model_class, X_train, y_train, X_val, y_val, hyperparameters):
    """
    Tune hyperparameters of a classification model using grid search and evaluate its performance on validation data.

    Parameters:
        - model_class (class): The classification model class to be tuned.
        - X_train (array-like): Training data features.
        - y_train (array-like): Training data labels.
        - X_val (array-like): Validation data features.
        - y_val (array-like): Validation data labels.
        - hyperparameters (dict): Hyperparameter grid for grid search.

    Returns:
        - best_estimator: The best tuned classification model.
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
        - folder (str): The folder where the files will be saved.
    Returns:
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

def evaluate_all_models(X_train_reg, y_train_reg, X_val_reg, y_val_reg, regression_hyperparameters,
                        X_train_cls, y_train_cls, X_val_cls, y_val_cls, classification_hyperparameters,
                        regression_models, classification_models):
    """
    This function performs hyperparameter tuning and evaluation for multiple regression and classification models
    using the provided training and validation datasets. It iterates over a list of regression and classification models,
    their corresponding hyperparameters, tunes each model, evaluates its performance on the validation dataset,
    and saves the best model, hyperparameters, and performance metrics for both regression and classification models.

    Parameters:
        - X_train_reg (array-like): Training data features for regression.
        - y_train_reg (array-like): Training data labels for regression.
        - X_val_reg (array-like): Validation data features for regression.
        - y_val_reg (array-like): Validation data labels for regression.
        - regression_hyperparameters (list of dict): List of hyperparameters for different regression models.
        - X_train_cls (array-like): Training data features for classification.
        - y_train_cls (array-like): Training data labels for classification.
        - X_val_cls (array-like): Validation data features for classification.
        - y_val_cls (array-like): Validation data labels for classification.
        - classification_hyperparameters (list of dict): List of hyperparameters for different classification models.
        - regression_models (list of regressor models): List of regression model instances.
        - classification_models (list of classifier models): List of classification model instances.

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
        save_model(best_estimator, best_params, performance_metrics, folder = "models/regression")
    
    # Iterate over the models and hyperparameters and perform tuning and evaluation for classification models
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
        save_model(best_estimator, best_params, performance_metrics, folder = "models/classification"
)

def find_best_model(X_train_reg, y_train_reg, X_val_reg, y_val_reg, regression_hyperparameters,
                    X_train_cls, y_train_cls, X_val_cls, y_val_cls, classification_hyperparameters,
                    regression_models, classification_models):
    """
    Find the best regression and classification models among a list of models by evaluating their performance on
    validation data.

    Parameters:
        - X_train_reg (array-like): Training data features for regression.
        - y_train_reg (array-like): Training data labels for regression.
        - X_val_reg (array-like): Validation data features for regression.
        - y_val_reg (array-like): Validation data labels for regression.
        - regression_hyperparameters (list of dict): List of hyperparameters for different regression models.
        - X_train_cls (array-like): Training data features for classification.
        - y_train_cls (array-like): Training data labels for classification.
        - X_val_cls (array-like): Validation data features for classification.
        - y_val_cls (array-like): Validation data labels for classification.
        - classification_hyperparameters (list of dict): List of hyperparameters for different classification models.
        - regression_models (list of regressor models): List of regression model instances.
        - classification_models (list of classifier models): List of classification model instances.

    Returns:
        - best_model_reg: The best trained regression model.
        - best_hyperparams_reg (dict): The best hyperparameters for regression.
        - best_performance_metrics_reg (dict): Performance metrics for the best regression model.
        - best_model_cls: The best trained classification model.
        - best_hyperparams_cls (dict): The best hyperparameters for classification.
        - best_performance_metrics_cls (dict): Performance metrics for the best classification model.
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

    # Define a dictionary of hyperparameters for different classification models
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
    {
        'model_name': 'LogisticRegression',
        'hyperparameters': {
            'penalty': ['None', 'l2'],
            'C': [0.1, 1],
            'solver': ['lbfgs', 'saga'],
            'max_iter': [100, 500, 1000]

        }
    }
    # {
    #     'model_name': 'SGDClassifier',
    #     'hyperparameters': {
    #         'loss': ['hinge', 'log', 'modified_huber'],
    #         'penalty': ['l2', 'l1', 'elasticnet'],
    #         'max_iter': [1000, 2000],
    #         'alpha': [0.0001, 0.001, 0.01]    
    #         }
    # }
]

    # Define a list of all the regression models
    regression_models = [GradientBoostingRegressor(), SVR(), DecisionTreeRegressor(), RandomForestRegressor(), SGDRegressor()]

    # Define a list of all the classification models
    classification_models = [GradientBoostingClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), LogisticRegression()]
        
    evaluate_all_models(X_train_reg, y_train_reg, X_val_reg, y_val_reg, regression_hyperparameters,
                            X_train_cls, y_train_cls, X_val_cls, y_val_cls, classification_hyperparameters,
                            regression_models, classification_models)


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