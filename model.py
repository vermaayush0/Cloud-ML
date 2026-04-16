# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("C:/Users/ayush/Downloads/Cloud/Iris.csv")
# Drop unnecessary column (if exists)
if 'Id' in df.columns:
    df = df.drop(columns=['Id'])

# Features and target
X = df.drop(columns=['Species'])
y = df['Species']

# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", accuracy)

# Test with custom input
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)

print("🌸 Prediction:", prediction)