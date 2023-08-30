from sklearn.linear_model import SGDRegressor 
from sklearn.model_selection import train_test_split
from tabular_data import load_airbnb
import pandas as pd

file_path = '/Users/andreasyianni/Desktop/Educational Training /Data Scientist Training/AiCore Training/Projects/Modelling-Airbnbs-Property-Listing-Dataset/airbnb-property-listings/tabular_data/clean_tabular_data.csv'

data = load_airbnb(file_path, "Price_Night")

encoded_data = pd.get_dummies(data, columns=['ID', 'Category'])
X = encoded_data.drop(columns=['Price_Night'])
y = encoded_data['Price_Night']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SGDRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
