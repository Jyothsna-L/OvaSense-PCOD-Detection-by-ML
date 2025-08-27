import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv("C:/Users/Admin/Desktop/pcod.csv")

df["Unusual_Bleeding"] = df["Unusual_Bleeding"].map({"yes": 1, "no": 0})

X = df[["Age", "Height", "Weight", "Unusual_Bleeding"]].apply(pd.to_numeric, errors="coerce").fillna(0)
y = np.where((X["Unusual_Bleeding"] == 1) | (X["Age"] > 30), 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

with open("pcod_model.pkl", "wb") as f:
    pickle.dump((svm_model, scaler), f)

print("Model and scaler saved as pcod_model.pkl")
