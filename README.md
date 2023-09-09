# **Modelling-Airbnbs-Property-Listing-Dataset**
# Table of Contents
1. [Introduction](#introduction)
2. [Data Preperation](#section-1)
3. [Machine Learning Models](#section-2)
    - [Regression & Classification Models 1.1](#subsection-1.1)
    - [Deep Neural Network 1.2](#subsection-1.2)


## Introduction
###### Remember to revise this as you go through project milestones
The devised framework for model development systematically addresses various tasks undertaken by the Airbnb team. Beginning with precise task definitions and data collection, the framework encompasses feature engineering and preprocessing, catering to tabular, text, and image data. Model architectures are thoughtfully selected, followed by rigorous hyperparameter tuning to optimize performance. Models are trained with careful consideration of overfitting prevention, and validation metrics inform the choice of model architecture. Ensemble techniques are explored for added robustness. Interpretability tools shed light on model predictions, aiding decision-making. Final models are rigorously evaluated against test data before deployment, complemented by continuous monitoring for maintenance. Comprehensive documentation and effective communication with stakeholders facilitate collaboration. This holistic approach ensures the deployment of effective, interpretable, and adaptable models aligned with Airbnb's data-driven practices.

## Data Preperation
We're working with Airbnb tabular data to create a comprehensive processing framework. To handle the data, we begin by creating a `tabular_data.py` script. We define the `remove_rows_with_missing_ratings` function to eliminate rows with missing rating values, enhancing data quality. For the "Description" column, we implement `combine_description_strings` to join list items into coherent strings, eliminating empty quotes and the "About this space" prefix. Next, the `set_default_feature_values` function fills empty "guests", "beds", "bathrooms", and "bedrooms" entries with 1, preserving the data's integrity. All these functions are integrated into `clean_tabular_data`, a comprehensive function that takes raw data as input and applies these steps sequentially. Within an `if __name__ == "__main__"` block, we load the raw data, call `clean_tabular_data` on it, and save the processed data as `clean_tabular_data.csv`, maintaining the file structure. Fianlly, we define a function called `load_airbnb` which returns the features and labels of the data in a tuple like (features, labels).

## Machine Learning Models

### Regression and Classification Models
The `modelling.py` script serves as a robust and comprehensive tool for the development, optimization, and evaluation of machine learning models tailored for both regression and classification tasks using scikit-learn. It commences the process by importing the dataset via the `load_airbnb` function, skillfully dividing it into distinct segments for regression (with a focus on predicting "Price_Night") and classification (aiming to predict "Category"), and then proceeds to train a diverse array of models.

Within this framework, the script orchestrates a systematic and meticulous evaluation of numerous regression and classification models, leveraging the versatile `evaluate_all_models` function. Notably, it employs the sophisticated `GridSearchCV` technique within the `tune_regression_model_hyperparameters` and `tune_classification_model_hyperparameters` functions to finely tune hyperparameters for each model. Throughout this evaluation journey, crucial performance metrics are diligently computed to gauge model effectiveness.

Upon completion of the model evaluations, the script utilizes the `save_model` function to preserve the optimal model estimators, along with their corresponding hyperparameters and pertinent performance metrics, in a local file repository. Finally, through the `find_best_model` function, the script identifies the most adept regression and classification models, offering detailed insights into their optimal hyperparameters and performance metrics.

By automating these intricate procedures, the script significantly enhances the efficiency of model selection for the prediction of Airbnb property prices per night and property categories.

### Deep Neural network


