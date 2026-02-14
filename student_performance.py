import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# Create Sample Dataset
# -----------------------------
np.random.seed(42)

data_size = 200

data = pd.DataFrame({
    "study_hours": np.random.randint(1, 10, data_size),
    "attendance": np.random.randint(50, 100, data_size),
    "previous_marks": np.random.randint(40, 100, data_size),
})

# Create target variable (Pass=1, Fail=0)
data["result"] = (
    (data["study_hours"] * 5 +
     data["attendance"] * 0.3 +
     data["previous_marks"] * 0.5) > 100
).astype(int)

# -----------------------------
# Split Data
# -----------------------------
X = data[["study_hours", "attendance", "previous_marks"]]
y = data["result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------
# Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)

print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# -----------------------------
# User Input Prediction
# -----------------------------
print("\n--- Student Performance Prediction ---")
study_hours = float(input("Enter Study Hours (1-10): "))
attendance = float(input("Enter Attendance (%): "))
previous_marks = float(input("Enter Previous Marks (0-100): "))

input_data = np.array([[study_hours, attendance, previous_marks]])
prediction = model.predict(input_data)

if prediction[0] == 1:
    print("Prediction: PASS ✅")
else:
    print("Prediction: FAIL ❌")
