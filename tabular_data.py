import ast
import os
import pandas as pd

def remove_rows_with_missing_ratings(dataframe):
    """
    Remove rows with missing rating values from the DataFrame.

    Parameters:
            dataframe (pandas.DataFrame): The input DataFrame containing rating columns.

    Returns:
            dataframe (pandas.DataFrame): A DataFrame with rows containing missing rating values removed.
    """
    columns_for_cleaning = ['Cleanliness_rating', 
                            'Accuracy_rating', 
                            'Location_rating', 
                            'Check-in_rating', 
                            'Value_rating', 
                            'Communication_rating']
    df_cleaned = dataframe.dropna(subset=columns_for_cleaning)

    return df_cleaned 

def preprocess_description(description):
    """
    This function takes a string representation of a list containing descriptions, it then processes and combines them and 
    removing empty strings and a prefix.

    Parameters:
            description (str):  A string containing a list-like representation of descriptions.

    Returns:
            combined_description (str): The combined and cleaned description.
    """
    try:
        # Convert string to a list
        description_list = ast.literal_eval(description)
        # Remove empty strings
        description_list = [item for item in description_list if item.strip() != '']
        # Remove prefix and extra whitespace
        description_list = [item.replace("About this space", "").strip() for item in description_list]
        # Join cleaned descriptions with a space
        combined_description = " ".join(description_list)

        return combined_description
    
    except (ValueError, SyntaxError):

        return None

def combine_description_strings(dataframe):
    """
    This function applies the 'preprocess_description' function to the 'Description'
    column of the given DataFrame, removing rows with missing descriptions.

    Parameters:
            dataframe (pandas.DataFrame): The DataFrame containing the 'Description' column.
        
    Returns:
            dataframe (pandas.DataFrame): The modified DataFrame with cleaned and combined descriptions.
    """
    dataframe['Description'] = dataframe['Description'].apply(preprocess_description)
    dataframe = dataframe.dropna(subset=['Description'])

    return dataframe

def set_default_feature_values(dataframe):
    """    
    This function takes a DataFrame and replaces empty values in the columns
    "guests", "beds", "bathrooms", and "bedrooms" with the number 1.
    
    Parameters:
        dataframe (pandas.DataFrame): The DataFrame containing the specified columns.
        
    Returns:
        dataframe (pandas.DataFrame): The modified DataFrame with replaced values.
    """
    columns_for_cleaning = ['guests', 
                            'beds', 
                            'bathrooms', 
                            'bedrooms' 
                            ]
    for column in columns_for_cleaning:
        dataframe[column].fillna(1, inplace=True)

    return dataframe

def clean_tabular_data(dataframe):
    """    
    This function takes a raw DataFrame and applies sequential processing steps
    to clean and modify the data.
    
    Parameters:
        dataframe (pandas.DataFrame): The raw DataFrame to be processed.
        
    Returns:
        df_proceessed (pandas.DataFrame): The processed DataFrame after all cleaning steps.
    """
    df_cleaned_ratings = remove_rows_with_missing_ratings(dataframe)
    df_cleaned_descriptions = combine_description_strings(df_cleaned_ratings)
    df_processed = set_default_feature_values(df_cleaned_descriptions)

    return df_processed  

def load_airbnb(file_path, label):
    """
    Load features and labels from the Airbnb tabular data.

    Parameters:
        file_path (str): Path to the CSV file containing the tabular data.
        label (str): Name of the column to be used as the label.

    Returns:
        tuple: A tuple containing a pandas DataFrame of features and a pandas Series of labels.
    """
    data = pd.read_csv(file_path)
    selected_columns =['guests', 'beds', 'bathrooms','Cleanliness_rating',
                        'Accuracy_rating', 'Communication_rating', 'Location_rating',
                        'Check-in_rating', 'Value_rating', 'amenities_count', 'bedrooms']
    features = data[selected_columns]
    labels = data[label]

    return features, labels

if __name__ == "__main__":

    csv_file_path = '/Users/andreasyianni/Desktop/Educational Training /Data Scientist Training/AiCore Training/Projects/Modelling-Airbnbs-Property-Listing-Dataset/airbnb-property-listings/tabular_data/listing.csv'
    file_path = '/Users/andreasyianni/Desktop/Educational Training /Data Scientist Training/AiCore Training/Projects/Modelling-Airbnbs-Property-Listing-Dataset/airbnb-property-listings/tabular_data/clean_tabular_data.csv'

    # Open the CSV file using a context manager and read it into a DataFrame
    with open(csv_file_path, 'r') as file:
        df = pd.read_csv(file)

    processed_data = clean_tabular_data(df)

    # Get the directory of the raw CSV file
    directory = os.path.dirname(csv_file_path)

    # Path to save the processed data CSV file
    processed_csv_file_path = os.path.join(directory, 'clean_tabular_data.csv')

    # Save the processed data as CSV
    processed_data.to_csv(processed_csv_file_path, index=False)
    print("Data processing and saving complete")