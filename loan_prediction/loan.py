import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ===============================
# 📂 Load and preprocess dataset
# ===============================
df = pd.read_csv("loan.csv")

# Fill missing values
df["Gender"].fillna(df["Gender"].mode()[0], inplace=True)
df["Married"].fillna(df["Married"].mode()[0], inplace=True)
df["Dependents"].fillna(df["Dependents"].mode()[0], inplace=True)
df["Self_Employed"].fillna(df["Self_Employed"].mode()[0], inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0], inplace=True)

# Handle outliers (capping)
cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
for col in cols:
    q1, q2, q3 = np.percentile(df[col], [25, 50, 75])
    IQR = q3 - q1
    if col == 'LoanAmount':
        lower_limit = q1 - 3 * IQR
        upper_limit = q3 + 3 * IQR
    else:
        lower_limit = q1 - 1.5 * IQR
        upper_limit = q3 + 1.5 * IQR
    df[col] = np.where(df[col] > upper_limit, upper_limit,
                       np.where(df[col] < lower_limit, lower_limit, df[col]))

# Encoding categorical features
df["Gender"] = pd.get_dummies(df["Gender"], dtype=int, drop_first=True)
df["Married"] = pd.get_dummies(df["Married"], dtype=int, drop_first=True)

# Label encoding for multiple columns

le_dependents = LabelEncoder()
le_education = LabelEncoder()
le_self_employed = LabelEncoder()
le_property_area = LabelEncoder()

df["Dependents"] = le_dependents.fit_transform(df["Dependents"])
df["Education"] = le_education.fit_transform(df["Education"])
df["Self_Employed"] = le_self_employed.fit_transform(df["Self_Employed"])
df["Property_Area"] = le_property_area.fit_transform(df["Property_Area"])
df["Loan_Status"] = LabelEncoder().fit_transform(df["Loan_Status"])


# Drop unnecessary column
df.drop(columns=['Loan_ID'], inplace=True)

# Split features and target
X = df.drop(["Loan_Status"], axis=1)
y = df["Loan_Status"]

# ===============================
# 📊 Train-test split + Scaling + SMOTE
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# ===============================
# 🧠 Train Logistic Regression Model
# ===============================
model = LogisticRegression(
    max_iter=1000, 
    random_state=42, 
    solver='lbfgs', 
    C=1.0
)
model.fit(X_train_bal, y_train_bal)

# ===============================
# 🎯 Evaluate Model
# ===============================
y_pred_bal = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_bal)
print(f"\n===============================")
print(f"🧠 Model: Logistic Regression")
print(f"===============================")
print(f"Accuracy after SMOTE: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_bal))

# Save model
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scaler
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save label encoders separately
pickle.dump(le_dependents, open("le_dependents.pkl", "wb"))
pickle.dump(le_education, open("le_education.pkl", "wb"))
pickle.dump(le_self_employed, open("le_self_employed.pkl", "wb"))
pickle.dump(le_property_area, open("le_property_area.pkl", "wb"))

print("\n✅ Model and encoders saved successfully!")





