# **Modelling-Airbnbs-Property-Listing-Dataset**
# Table of Contents
1. [Introduction](#introduction)
2. [Data Preperation](#section-1)
    - [Subsection 1.1](#subsection-1.1)
    - [Subsection 1.2](#subsection-1.2)


## Introduction
The devised framework for model development systematically addresses various tasks undertaken by the Airbnb team. Beginning with precise task definitions and data collection, the framework encompasses feature engineering and preprocessing, catering to tabular, text, and image data. Model architectures are thoughtfully selected, followed by rigorous hyperparameter tuning to optimize performance. Models are trained with careful consideration of overfitting prevention, and validation metrics inform the choice of model architecture. Ensemble techniques are explored for added robustness. Interpretability tools shed light on model predictions, aiding decision-making. Final models are rigorously evaluated against test data before deployment, complemented by continuous monitoring for maintenance. Comprehensive documentation and effective communication with stakeholders facilitate collaboration. This holistic approach ensures the deployment of effective, interpretable, and adaptable models aligned with Airbnb's data-driven practices.

## Data Preperation
We're working with Airbnb tabular data to create a comprehensive processing framework. To handle the data, we begin by creating a tabular_data.py script. We define the remove_rows_with_missing_ratings function to eliminate rows with missing rating values, enhancing data quality. For the "Description" column, we implement combine_description_strings to join list items into coherent strings, eliminating empty quotes and the "About this space" prefix. Next, the set_default_feature_values function fills empty "guests", "beds", "bathrooms", and "bedrooms" entries with 1, preserving the data's integrity. All these functions are integrated into clean_tabular_data, a comprehensive function that takes raw data as input and applies these steps sequentially. Finally, within an if __name__ == "__main__" block, we load the raw data, call clean_tabular_data on it, and save the processed data as clean_tabular_data.csv, maintaining the file structure."

### Subsection 1.1
Content here...

### Subsection 1.2
Content here...

