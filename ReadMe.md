docker build -t dockerfile:v1 .

docker run dockerfile:v1

# For KNN model and heart disease dataset run

docker build -t knn-heart-disease .
docker run knn-heart-disease
