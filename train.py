import os
import pickle

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Chargement du dataset
print("==> Chargement du dataset Breast Cancer...")
data = load_breast_cancer()
X, y = data.data, data.target
print(f"    Taille : {X.shape[0]} exemples, {X.shape[1]} features")
print(f"    Classes : {list(data.target_names)}")

# 2. Prétraitement
print("\n==> Prétraitement...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Entraînement
print("\n==> Entraînement du modèle...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Évaluation
print("\n==> Évaluation...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n    Accuracy : {acc:.4f}")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# 5. Sauvegarde
os.makedirs("artifacts", exist_ok=True)

artifact = {
    "model": model,
    "scaler": scaler,
    "feature_names": list(data.feature_names),
    "target_names": list(data.target_names),
}

with open("artifacts/model.pkl", "wb") as f:
    pickle.dump(artifact, f)

print("\n==> Modèle sauvegardé dans artifacts/model.pkl")
