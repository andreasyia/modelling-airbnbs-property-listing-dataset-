from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, precision_score , recall_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from tabular_data import load_airbnb

file_path = '/Users/andreasyianni/Desktop/Educational Training /Data Scientist Training/AiCore Training/Projects/Modelling-Airbnbs-Property-Listing-Dataset/airbnb-property-listings/tabular_data/clean_tabular_data.csv'

# Load the Airbnb data and split it into features (X) and target (y)
data = load_airbnb(file_path, 'Category')
X, y = data

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LogisticRegression model and train it
model = LogisticRegression(solver='liblinear', max_iter=1000)
model.fit(X_train, y_train)

# Make predictions using the trained model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate performance metrics for the training set
train_f1_score = f1_score(y_train, y_train_pred, average='macro')
train_precision = precision_score(y_train, y_train_pred, average='macro')
train_recall = recall_score(y_train, y_train_pred, average='macro')
train_accuracy = accuracy_score(y_train, y_train_pred)

# Calculate performance metrics for the test set
test_f1_score = f1_score(y_test, y_test_pred, average='macro')
test_precision = precision_score(y_test, y_test_pred, average='macro')
test_recall = recall_score(y_test, y_test_pred, average='macro')
test_accuracy = accuracy_score(y_test, y_test_pred)

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

def tune_classification_model_hyperparameters(model_class, X_train, y_train, X_test, y_test, hyperparameters):
    
    model = model_class(**hyperparameters)
    model.fit(X_train, y_train)
    model.predict(X_test)

    grid = GridSearchCV(model_class, param_grid=hyperparameters, refit = True, verbose = 3,n_jobs=-1)
    grid.fit(X_train, y_train)

    best_estimator = grid.best_estimator_
    best_params = grid.best_params_

    val_predictions = best_estimator.predict(X_test)
    val_accuracy = accuracy_score(y_test, val_predictions)

    performance_metrics = {"validation_accuracy": val_accuracy}

    return best_estimator, best_params, performance_metrics


hyperparameters_lr = {'C': [0.1],
                      'penalty': ['l1'],
                      'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
                      }

tune_classification_model_hyperparameters(
    model_class= LogisticRegression,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    hyperparameters=hyperparameters_lr
    )





