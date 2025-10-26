from sklearn.datasets import load_wine  
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, classification_report  
import joblib

if __name__ == '_main_':
    # Load the Wine dataset
    wine = load_wine()
    X, y = wine.data, wine.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Logistic Regression classifier
    model = LogisticRegression(max_iter=200)  # Adjusted hyperparameter
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save the trained model to a file
    joblib.dump(model, 'wine_model.pkl')  # Changed filename to reflect new model
    
    print("The model training was successful and saved as 'wine_model.pkl'")