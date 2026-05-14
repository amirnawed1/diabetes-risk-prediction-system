# train_model.py
# Train and save the diabetes prediction model for Streamlit deployment

import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("diabetes.csv")

# Split features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split. Stratify keeps diabetic/non-diabetic ratio balanced.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

# Pipeline = scaling + model together, so no separate scaler file is needed.
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train model
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", round(accuracy, 4))
print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))
print("Classification Report:\n", classification_report(y_test, predictions))

# Save model
with open("diabetes_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as diabetes_model.pkl")
