import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print("Starting model training...")
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(' ', np.nan))
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

y = (df['Churn'] == 'Yes').astype(int)
X = df.drop(columns=['customerID', 'Churn'])

cat_cols = [c for c in X.columns if X[c].dtype == 'object']
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
feature_names = X_encoded.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

os.makedirs('models', exist_ok=True)
with open('models/best_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('models/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

print("SUCCESS: Trained and saved best_model.pkl, scaler.pkl, feature_names.pkl")
