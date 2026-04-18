''''import pandas as pd
#data = pd.read_csv("HeartDiseaseTrain-Test.csv")
#print(data.head())
#print(data.info())
#print(data.describe())


# Load dataset
data = pd.read_csv("HeartDiseaseTrain-Test.csv")
data = pd.get_dummies(data, drop_first=True)

# Show first 5 rows
print("First 5 rows of dataset:")
print(data.head())

# Show dataset info
print("\nDataset Info:")
print(data.info())

# Show basic statistics
print("\nDataset Description:")
print(data.describe())

# Show columns
print("Columns in dataset:")
print(data.columns)

# Check missing values
print("\nMissing values:")
print(data.isnull().sum())

# Basic statistics
print("\nDataset Description:")
print(data.describe())
print(data.columns)
# Features (input)
X = data.drop("target", axis=1)

# Target (output)
y = data["target"]

print(X.head())
print(y.head())
X = data.drop("target", axis=1)

# Output (what we predict)
y = data["target"]

print("X shape:", X.shape)
print("y shape:", y.shape)
from sklearn.model_selection import train_test_split

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
from sklearn.ensemble import RandomForestClassifier

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X_train, y_train)

print("Model training complete")'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("HeartDiseaseTrain-Test.csv")

# Convert text to numbers
data = pd.get_dummies(data, drop_first=True)

# Split data
X = data.drop("target", axis=1)
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "heart_model.pkl")


