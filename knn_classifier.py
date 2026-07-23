import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier

wineData = load_wine()

# Features
X = wineData.data

# Labels
y = wineData.target

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

#create mode with k=5
model= KNeighborsClassifier(n_neighbors=5)

#Train the model
model.fit(X_train,y_train)

#predict on test datasets
predictions=model.predict(X_test)

print("Predicted classes:")
print(predictions)

print("\nActual classes:")
print(y_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"\nAccuracy: {accuracy:.2%}")
'''
# Confusion Matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

# Display Confusion Matrix
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=wineData.target_names
)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
'''