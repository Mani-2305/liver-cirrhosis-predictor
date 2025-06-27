
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load training data
X_train = pd.read_csv("Data/X_train.csv")
y_train = pd.read_csv("Data/y_train.csv")

# Flatten y_train from dataframe to array
y_train = y_train.values.ravel()

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate on training data
y_pred = model.predict(X_train)
accuracy = accuracy_score(y_train, y_pred)
print("✅ Random Forest Accuracy on training data:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n", classification_report(y_train, y_pred))

# Save model
with open("Flask/rf_stage.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved as rf_stage.pkl")
