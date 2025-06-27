
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("Data/liver_cirrhosis.csv")

# Drop rows with missing target (Status)
df = df[df['Status'].notnull()]

# Encode target variable: C=0, CL=1, D=2
status_mapping = {'C': 0, 'CL': 1, 'D': 2}
df['Status'] = df['Status'].map(status_mapping)

# Drop columns not useful for prediction
df.drop(columns=['Drug'], inplace=True)  # optionally drop 'Drug' if irrelevant

# Convert categorical columns
df['Sex'] = df['Sex'].map({'M': 1, 'F': 0})
df['Ascites'] = df['Ascites'].map({'Y': 1, 'N': 0})
df['Hepatomegaly'] = df['Hepatomegaly'].map({'Y': 1, 'N': 0})
df['Spiders'] = df['Spiders'].map({'Y': 1, 'N': 0})
df['Edema'] = df['Edema'].map({'Y': 1, 'N': 0})

# Drop rows with missing values (or optionally fill them)
df.dropna(inplace=True)

# Separate features and target
X = df.drop(columns=['Status'])
y = df[['Status']]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler
with open("Flask/normalizer.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Save split data for modeling
pd.DataFrame(X_train).to_csv("Data/X_train.csv", index=False)
pd.DataFrame(y_train).to_csv("Data/y_train.csv", index=False)
