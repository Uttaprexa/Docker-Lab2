# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Step 1: Load the Heart Disease dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Define column names
column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Load dataset
data = pd.read_csv(url, names=column_names)

# Step 2: Preprocessing

# Replace '?' with NaN and drop rows with missing values
data.replace('?', np.nan, inplace=True)
data.dropna(inplace=True)

# Convert to numeric types
for col in ["ca", "thal"]:
    data[col] = pd.to_numeric(data[col])

# Convert target to binary classification (0: no disease, 1: has disease)
data["target"] = data["target"].apply(lambda x: 1 if int(x) > 0 else 0)

# Features and target
X = data.drop("target", axis=1)
y = data["target"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model and hyperparameter tuning with GridSearchCV
param_grid = {
    'n_neighbors': list(range(3, 21)),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]  # 1 = Manhattan, 2 = Euclidean
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_knn = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Step 5: Evaluation
y_pred = best_knn.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Step 6: Cross-validation
cv_scores = cross_val_score(best_knn, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))

# Step 7: Save the model
joblib.dump(best_knn, "heart_disease_knn_model.pkl")
print("\nModel saved as 'heart_disease_knn_model.pkl'")