# **Modelling-Airbnbs-Property-Listing-Dataset**
# Table of Contents
1. [Introduction](#introduction)
2. [Data Preperation](#section-1)
3. [Machine Learning Models](#section-2)
    - [Regression Model 2.1](#subsection-2.1)
    - [Classification Model 2.2](#subsection-2.2)
    - [Deep Neural Network Model 2.3](#subsection-2.3)
4. [Reuse the Framework for additional Airbnb Data analyses](#section-3)
5. [Conclusion](#section-4)
6. [Technologies Used](#section-5)
    - [Imported Modules 5.1](#subsection-5.1)
    - [Programming Languages 5.2](#subsection-5.2)
7. [License](#section-6)



## Introduction
This comprehensive suite of scripts and modules forms the backbone of a robust machine learning project centered around Airbnb property listings. The project encompasses multifaceted aspects, from meticulous data preparation and optimization to the fine-tuning and evaluation of diverse machine learning models. The core objective revolves around analyzing and predicting crucial attributes within Airbnb listings. Each script and module within this project plays a pivotal role, contributing specialized functions and classes tailored for distinct tasks. From data preprocessing using the `tabular_data.py` script to the evaluation of regression and classification models via `regression_model.py` and `classification_model.py` respectively, each element is crafted to streamline and enhance the predictive capabilities and analytical depth of the project. Additionally, the `neural_network.py` script presents a deep neural network-based regression model, delving into the intricate realm of nightly price prediction. Together, these components form a comprehensive framework primed for analyzing, predicting, and optimizing various facets of Airbnb property listings.

## Data Preperation
This `tabular_data.py` script tailored for Airbnb tabular data processing. It offers specialized functions to refine and optimize the dataset for analysis. The `remove_rows_with_missing_ratings` function bolsters data reliability by eliminating rows lacking rating values. `combine_description_strings` streamlines the "Description" column for enhanced readability. Additionally, `set_default_feature_values` fills void entries in crucial columns. The `pivotal clean_tabular_data` function orchestrates these steps, ensuring comprehensive data refinement. And lastly, `load_airbnb` utility simplifies data retrieval, packaging features and labels into a tuple for modeling tasks. This framework provides an end-to-end solution for enhancing Airbnb tabular data, priming it for advanced analytical pursuits.

## Machine Learning Models

### Regression Model
The `regression_model.py` script provided conducts evaluation and tuning of regression models using scikit-learn for predicting Airbnb property prices. It utilizes various models including `Gradient Boosting Regressor`, `SVR`, `Decision Tree Regressor`, `Random Forest Regressor` and `SGD Regressor` classifiers from scikit-learn.It loads the dataset, splits it into training and validation sets, and employs various regression models from scikit-learn, evaluating them based on Mean Squared Error (MSE) and R-squared (R2) metrics. To use this script, ensure you have dependencies like scikit-learn, NumPy, and joblib installed. Clone the repository, navigate to the script directory, and execute it using Python 3.x. This script contains functions to optimize regression models. `custom_tune_regression_model_hyperparameters` fine-tunes models by custom grid search, selecting the best based on the lowest validation RMSE. `save_model` stores trained models, their parameters, and validation metrics separately. `evaluate_all_models` manages model tuning, evaluation, and stores the best ones. `find_best_model` systematically optimizes and identifies the top-performing model, its parameters, and metrics. These functions streamline regression model improvement for better accuracy and efficiency.

### Classification Model
The `classification_model.py` script conducts evaluation and tuning of classification models using scikit-learn for predicting Airbnb property categories. It utilizes various models including `Logistic Regression`, `Gradient Boosting`, `Random Forest`, `Decision Tree` and `Stochastic Gradient Descent (SGD)` classifiers from scikit-learn. The script loads the Airbnb dataset, splits it into training and validation sets, and trains a Logistic Regression model. It computes performance metrics such as F1 Score, Precision, Recall, and Accuracy for both training and test sets. Ensure you have the required dependencies such as scikit-learn, NumPy, and joblib installed. Clone the repository and navigate to the script directory. Execute the script using Python 3.x. This script provides functions tailored for optimizing classification models. `tune_classification_model_hyperparameters` fine-tunes models using GridSearchCV, selecting the best based on given metrics. `save_model` preserves trained models, their parameters, and validation metrics separately. `evaluate_all_models` manages model tuning, evaluation, and stores the best ones. `find_best_model` systematically optimizes and identifies the best-performing model, its parameters, and metrics. These functions streamline classification model refinement for better accuracy and effectiveness.

### Deep Neural Network Model
The `neural_network.py` script consists of classes and functions for a neural network-based regression model applied to Airbnb nightly price prediction. The `AirbnbNightlyPriceRegressionDataset` class defines a PyTorch Dataset for regression tasks, providing access to features and labels. The `NN` class constructs a feedforward neural network for inference, initialized with configurable parameters. Several key functions are implemented: `get_nn_config` retrieves neural network configurations from a YAML file, `calculate_accuracy` computes accuracy metrics, `calculate_rmse` determines RMSE values, and `calculate_r_squared` calculates R-squared scores. The `train` function orchestrates model training, evaluating performance with TensorBoardX logging, while save_model() saves pertinent model-related data. Additionally, `generate_nn_configs` creates diverse network configurations, and `find_best_nn` conducts K-fold cross-validation to determine the optimal model based on RMSE. Each function is extensively documented via docstrings, clarifying their roles, parameters, and return values for improved comprehension and streamlined utilization.

Below there is the structured of Neural Network used in this model and screenshots taken from teensorboard of the model performance.

![image info](/pictures/nn.png)

![image info](/pictures/train_accuracy.png)

![image info](/pictures/train_rmse.png)

![image info](/pictures/val_accuracy.png)

![image info](/pictures/val_rmse.png)



## Reuse the Framework for additional Airbnb Data analyses
This machine learning project delves into analyzing Airbnb property listings, with a primary objective of predicting the count of bedrooms. A subset of a larger project, it uses a dataset encompassing features like guest count, beds, bathrooms, cleanliness ratings, etc. The task involves preprocessing data and engineering features to train models aimed at predicting the integer number of bedrooms. The project evaluates various machine learning models using metrics like accuracy and RMSE to automate bedroom prediction within Airbnb property listings.

Below there are screenshots taken from teensorboard of the model performance and accuracy.

![image info](/pictures/reuse_framework/train_accuracy.png)

![image info](/pictures/reuse_framework/train_rmse.png)

![image info](/pictures/reuse_framework/val_accuracy.png)

![image info](/pictures/reuse_framework/val_rmse.png)



## Conclusion 
In conclusion, this project delivers a robust framework tailored for the comprehensive analysis and prediction of essential attributes within Airbnb property listings. By employing specialized scripts and modules, it ensures meticulous data preparation, optimization, and model evaluation. From refining data integrity and readability through the `tabular_data.py` script to harnessing diverse regression, classification, and neural network-based models, this project offers a multifaceted approach to predicting bedroom counts, Airbnb property categories and nightly prices. The scripts functionalities, encapsulated in their tailored functions and classes, streamline the analytical pipeline, offering a holistic solution for modeling and refining Airbnb data.

As for potential improvements, expanding the scope of integrating more advanced neural network architectures could enrich the project's predictive capabilities. Additionally, incorporating ensemble techniques or exploring more extensive hyperparameter tuning strategies might further enhance the models' accuracy and robustness. Further efforts toward feature engineering, exploring more extensive datasets, or leveraging advanced natural language processing (NLP) techniques could also provide deeper insights into property descriptions and other unstructured data within the Airbnb listings. Lastly, optimizing the training pipelines for these models, such as leveraging distributed computing or cloud-based training for more extensive datasets, can significantly scale up the analysis and prediction capabilities of the project.

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

