import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import joblib

# Step 2: Load your CSV file
data = pd.read_csv("Campus_Selection.csv")

# Convert 'gender' to numeric
data['gender'] = data['gender'].map({'M': 1, 'F': 0})

# Convert categorical columns to dummies (ensure all are included)
categorical_cols = ['ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation', 'status']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Now all columns are numeric, correlation will work
print("\nCorrelation:\n", data.corr())


# Step 3: See the first 5 rows of your dataset
print(data.head())
# Step 4: Check column names
print("Columns in dataset:", data.columns)

# Step 5: Check data types of each column
print("\nData types:\n", data.dtypes)

# Step 6: Check for missing values
print("\nMissing values:\n", data.isnull().sum())
# Step 7: Drop rows with missing values (optional)
data = data.dropna()

# Step 8: Reset index after dropping rows
data = data.reset_index(drop=True)

# Step 9: Check the cleaned dataset
print("\nCleaned Data:")
print(data.head())
# Step 10: Count how many students are placed vs not placed
placement_counts = data['status_Placed'].value_counts()
print("\nPlacement counts:\n", placement_counts)

# Step 11: Find average scores (if you have columns like 'CGPA' or 'Marks')
if 'CGPA' in data.columns:
    print("\nAverage CGPA:", data['CGPA'].mean())

# Step 12: Check correlation between numeric columns
print("\nCorrelation:\n", data.corr())

import matplotlib.pyplot as plt
import seaborn as sns

# Placement counts bar chart
sns.countplot(x='status_Placed', data=data)
plt.title("Placement Counts")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
# Average CGPA by placement
if 'CGPA' in data.columns:
    print(data.groupby('status_Placed')['CGPA'].mean())

# Count of work experience vs placement
if 'workex_Yes' in data.columns:
    print(data.groupby('workex_Yes')['status_Placed'].value_counts())

# ------------------ Step: Prepare data for modeling ------------------
X = data.drop('status_Placed', axis=1)
y = data['status_Placed']

# Split dataset into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------ Step: Train Decision Tree ------------------
from sklearn.tree import DecisionTreeClassifier, plot_tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# ------------------ Step: Make predictions ------------------
y_pred = dt_model.predict(X_test)

# ------------------ Step: Evaluate model ------------------
from sklearn.metrics import accuracy_score, classification_report
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------ Step: Feature Importance ------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importance:\n", importance)

# Visualize feature importance
plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=importance)
plt.title("Feature Importance for Placement Prediction")
plt.show()

# ------------------ Step: Visualize Decision Tree ------------------
plt.figure(figsize=(20,10))
plot_tree(dt_model, feature_names=X.columns, class_names=['Not Placed','Placed'], filled=True, rounded=True)
plt.show()

# 1. Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# 2. Make predictions on test set
y_pred = dt_model.predict(X_test)

# 3. Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 4. Now you can predict a new student
# Paste the new_student code here

# Create new student input with all columns
new_student = pd.DataFrame(0, index=[0], columns=X.columns)

# Fill in actual values
new_student['gender'] = 1
new_student['ssc_p'] = 85
new_student['hsc_p'] = 90
new_student['degree_p'] = 88
new_student['etest_p'] = 75
new_student['workex_Yes'] = 1

# For specialization, use the exact column name from X.columns
# Check your columns with:
print(X.columns)

# Predict placement for this new student
prediction = dt_model.predict(new_student)
print("Placement Prediction:", "Placed" if prediction[0]==1 else "Not Placed")
# ------------------ Step: Save the trained model ------------------

joblib.dump(dt_model, 'campus_placement_model.pkl')
# --- Paste after saving model ---
# Function to safely get float input
def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input! Please enter a number.")

# Function to safely get int input with allowed values
def get_int(prompt, allowed_values):
    while True:
        try:
            val = int(input(prompt))
            if val in allowed_values:
                return val
            else:
                print(f"Please enter one of {allowed_values}")
        except ValueError:
            print("Invalid input! Please enter an integer.")

# Function to safely get specialisation input
def get_specialisation(prompt):
    while True:
        val = input(prompt).strip()
        if val in ['Mkt&HR', 'Mkt&Fin']:
            return val
        else:
            print("Invalid input! Enter 'Mkt&HR' or 'Mkt&Fin'.")
            
def predict_student(gender, ssc_p, hsc_p, degree_p, etest_p, workex, specialisation):
    student = pd.DataFrame(0, index=[0], columns=X.columns)
    student['gender'] = gender
    student['ssc_p'] = ssc_p
    student['hsc_p'] = hsc_p
    student['degree_p'] = degree_p
    student['etest_p'] = etest_p
    student['workex_Yes'] = workex

    # Specialisation column
    if 'specialisation_Mkt&HR' in X.columns:
        student['specialisation_Mkt&HR'] = 1 if specialisation == 'Mkt&HR' else 0

    pred = dt_model.predict(student)
    return "Placed" if pred[0] == 1 else "Not Placed"

ssc_p = get_float("Enter SSC Percentage: ")
hsc_p = get_float("Enter HSC Percentage: ")
degree_p = get_float("Enter Degree Percentage: ")
etest_p = get_float("Enter E-test Percentage: ")
gender = get_int("Enter Gender (1 for Male, 0 for Female): ", [0, 1])
workex = get_int("Work Experience (1 for Yes, 0 for No): ", [0, 1])
specialisation = get_specialisation("Enter Specialisation (Mkt&HR / Mkt&Fin): ")


result = predict_student(gender, ssc_p, hsc_p, degree_p, etest_p, workex, specialisation)
print("Placement Prediction:", result)
