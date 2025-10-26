# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the model training script into the container
COPY src/ .

# Install Scikit-Learn and joblib
RUN pip install -r requirements.txt

# Run the script when the container launches
# Change the below command to main2.py, if you want to try the wine dataset and with Logistic Regression ML model
CMD ["python", "main2.py"]
