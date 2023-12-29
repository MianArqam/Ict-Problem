
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the Iris dataset from 'iris.csv'
iris_data = pd.read_csv("iris.csv")

# Separate features (X) and labels (y)
X = iris_data.drop("species", axis=1)
y = iris_data["species"]

# Display dataset statistics
print("Iris Dataset Statistics:")
print(X.describe())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the number of neighbors (k) for kNN
k = 3

# Initialize and train the kNN model
knn_model = KNeighborsClassifier(n_neighbors=k)
knn_model.fit(X_train_scaled, y_train)

# Predict labels for the test set
y_pred = knn_model.predict(X_test_scaled)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print model evaluation metrics
print("\nModel Evaluation:")
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_rep)

# Assume 'new_data' contains new samples for prediction (features only, without labels)
# For example, you might have new_data as follows:
new_data = [
    [5.1, 3.5, 1.4, 0.2],  # Sample 1
    [6.3, 2.9, 5.6, 1.8],  # Sample 2
    # Add more samples here if needed
]

# Scale the new data using the same scaler used for training data
new_data_scaled = scaler.transform(new_data)

# Make predictions on the new data using the trained model
new_data_predictions = knn_model.predict(new_data_scaled)

# Display the predictions for the new data
print("Predictions for new data:")
for i, prediction in enumerate(new_data_predictions):
    print(f"Sample {i+1} belongs to class: {prediction}")

from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Initialize lists to store cross-validation scores and k values
k_values = list(range(1, 21))
cv_scores = []

# Calculate cross-validation scores for different k values
for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=5)
    cv_scores.append(scores.mean())

# Plotting the cross-validation scores for different k values
plt.figure(figsize=(8, 6))
plt.plot(k_values, cv_scores, marker='o', linestyle='-')
plt.title('Cross-Validation Scores for Different k values')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Cross-Validation Accuracy')
plt.xticks(k_values)  # Show all k values on x-axis
plt.grid(True)
plt.show()

"""# New section"""