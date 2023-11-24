# **Modelling-Airbnbs-Property-Listing-Dataset**
# Table of Contents
1. [Introduction](#introduction)
2. [Data Preperation](#section-1)
3. [Machine Learning Models](#section-2)
    - [Regression Model 2.1](#subsection-2.1)
    - [Classification Model 2.2](#subsection-2.2)
    - [Deep Neural Network Model 2.3](#subsection-2.3)
4. [Utilize the Framework for additional Airbnb Data analyses](#section-3)
5. [Conclusion](#section-4)
6. [Technologies Used](#section-5)
    - [Imported Modules 5.1](#subsection-5.1)
    - [Programming Languages 5.2](#subsection-5.2)
7. [License](#section-6)



## Introduction
###### Remember to revise this as you go through project milestones
The devised framework for model development systematically addresses various tasks undertaken by the Airbnb team. Beginning with precise task definitions and data collection, the framework encompasses feature engineering and preprocessing, catering to tabular, text, and image data. Model architectures are thoughtfully selected, followed by rigorous hyperparameter tuning to optimize performance. Models are trained with careful consideration of overfitting prevention, and validation metrics inform the choice of model architecture. Ensemble techniques are explored for added robustness. Interpretability tools shed light on model predictions, aiding decision-making. Final models are rigorously evaluated against test data before deployment, complemented by continuous monitoring for maintenance. Comprehensive documentation and effective communication with stakeholders facilitate collaboration. This holistic approach ensures the deployment of effective, interpretable, and adaptable models aligned with Airbnb's data-driven practices.

## Data Preperation
This `tabular_data.py` script tailored for Airbnb tabular data processing. It offers specialized functions to refine and optimize the dataset for analysis. The `remove_rows_with_missing_ratings` function bolsters data reliability by eliminating rows lacking rating values. `combine_description_strings` streamlines the "Description" column for enhanced readability. Additionally, `set_default_feature_values` fills void entries in crucial columns. The `pivotal clean_tabular_data` function orchestrates these steps, ensuring comprehensive data refinement. And lastly, `load_airbnb` utility simplifies data retrieval, packaging features and labels into a tuple for modeling tasks. This framework provides an end-to-end solution for enhancing Airbnb tabular data, priming it for advanced analytical pursuits.

## Machine Learning Models

### Regression Model
The `regression_model.py` script provided conducts evaluation and tuning of regression models using scikit-learn for predicting Airbnb property prices. It utilizes various models including `Gradient Boosting Regressor`, `SVR`, `Decision Tree Regressor`, `Random Forest Regressor` and `SGD Regressor` classifiers from scikit-learn.It loads the dataset, splits it into training and validation sets, and employs various regression models from scikit-learn, evaluating them based on Mean Squared Error (MSE) and R-squared (R2) metrics. To use this script, ensure you have dependencies like scikit-learn, NumPy, and joblib installed. Clone the repository, navigate to the script directory, and execute it using Python 3.x. This script contains functions to optimize regression models. `custom_tune_regression_model_hyperparameters` fine-tunes models by custom grid search, selecting the best based on the lowest validation RMSE. `save_model` stores trained models, their parameters, and validation metrics separately. `evaluate_all_models` manages model tuning, evaluation, and stores the best ones. `find_best_model` systematically optimizes and identifies the top-performing model, its parameters, and metrics. These functions streamline regression model improvement for better accuracy and efficiency.

### Classification Model
The `classification_model.py` script conducts evaluation and tuning of classification models using scikit-learn for predicting Airbnb property categories. It utilizes various models including `Logistic Regression`, `Gradient Boosting`, `Random Forest`, `Decision Tree` and `Stochastic Gradient Descent (SGD)` classifiers from scikit-learn. The script loads the Airbnb dataset, splits it into training and validation sets, and trains a Logistic Regression model. It computes performance metrics such as F1 Score, Precision, Recall, and Accuracy for both training and test sets. Ensure you have the required dependencies such as scikit-learn, NumPy, and joblib installed. Clone the repository and navigate to the script directory. Execute the script using Python 3.x. This script provides functions tailored for optimizing classification models. `tune_classification_model_hyperparameters` fine-tunes models using GridSearchCV, selecting the best based on given metrics. `save_model` preserves trained models, their parameters, and validation metrics separately. `evaluate_all_models` manages model tuning, evaluation, and stores the best ones. `find_best_model` systematically optimizes and identifies the best-performing model, its parameters, and metrics. These functions streamline classification model refinement for better accuracy and effectiveness.

### Deep Neural Network Model
![image info](/pictures/nn.png)
The `neural_network.py` script consists of classes and functions for a neural network-based regression model applied to Airbnb nightly price prediction. The `AirbnbNightlyPriceRegressionDataset` class defines a PyTorch Dataset for regression tasks, providing access to features and labels. The `NN` class constructs a feedforward neural network for inference, initialized with configurable parameters. Several key functions are implemented: `get_nn_config` retrieves neural network configurations from a YAML file, `calculate_accuracy` computes accuracy metrics, `calculate_rmse` determines RMSE values, and `calculate_r_squared` calculates R-squared scores. The `train` function orchestrates model training, evaluating performance with TensorBoardX logging, while save_model() saves pertinent model-related data. Additionally, `generate_nn_configs` creates diverse network configurations, and `find_best_nn` conducts K-fold cross-validation to determine the optimal model based on RMSE. Each function is extensively documented via docstrings, clarifying their roles, parameters, and return values for improved comprehension and streamlined utilization.

## Utilize the Framework for additional Airbnb Data analyses

## Conclusion 

## Technologies Used

### Imported Modules
- ast
- datetime
- itertools
- joblib
- JSON
- math
- NumPy 
- os
- pandas
- PyTorch
- scikit-learn
- YAML

### Programming Languages
- Python

## License
MIT License

Copyright (c) [2023] [Andreas Yianni]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

