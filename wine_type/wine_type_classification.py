import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
wine_df = pd.read_csv("wine-quality-white-and-red.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
wine_df['type_encoded'] = le.fit_transform(wine_df['type'])

X = wine_df.drop(['type', 'type_encoded'], axis=1)
y = wine_df['type_encoded']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_bal, y_train_bal)

from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Average accuracy:", cv_scores.mean())

import pickle
# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("✅ Scaler saved successfully as 'scaler.pkl'")

# Save label encoder
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(le, file)

print("✅ Label encoder saved successfully as 'label_encoder.pkl'")



# Save the trained model
with open('wine_type_logistic_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model saved successfully as 'wine_type_logistic_model.pkl'")