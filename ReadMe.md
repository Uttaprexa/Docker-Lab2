docker build -t dockerfile:v1 .

docker run dockerfile:v1

# For KNN model and heart disease dataset run

docker build -t knn-heart-disease .
docker run knn-heart-disease


# Changes made:
- main.py has a KNN model, a heart disease dataset, hyperparameter tuning with GridSearchCV, model evaluation, and cross-validation
- main2.py has Logistic regression ML Model and a wine dataset.

Changes are made in the readme file, dockerfile and requirements accordingly.
