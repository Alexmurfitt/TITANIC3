# train_model.py

import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import optuna

warnings.filterwarnings("ignore")

# === 1. CARGA Y PREPROCESADO ===

train = pd.read_csv("train_clean.csv")   # Si prefieres usar los limpios
test = pd.read_csv("test_clean.csv")

print("Columnas en train:", train.columns)
print("Columnas en test:", test.columns)

X = train.drop("Survived", axis=1)
y = train["Survived"]

X_test = test.copy()

# === 2. ESCALADO DE VARIABLES ===

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.joblib")

# === 3. OPTIMIZACIÓN DE HIPERPARÁMETROS CON OPTUNA (LightGBM) ===

def objective_lgb(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "n_estimators": 300,
        "random_state": 42
    }
    model = LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy").mean()
    return score

print("🔎 Optimizando hiperparámetros de LightGBM...")
study_lgb = optuna.create_study(direction="maximize")
study_lgb.optimize(objective_lgb, n_trials=30)
best_params_lgb = study_lgb.best_params
best_params_lgb.update({"n_estimators": 300, "random_state": 42})
print("Mejores hiperparámetros LightGBM:", best_params_lgb)

# === 4. LO MISMO PARA XGBOOST (puedes ajustar los rangos según tu caso) ===

def objective_xgb(trial):
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "n_estimators": 300,
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }
    model = XGBClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    score = cross_val_score(model, X_scaled, y, cv=cv, scoring="accuracy").mean()
    return score

print("🔎 Optimizando hiperparámetros de XGBoost...")
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=30)
best_params_xgb = study_xgb.best_params
best_params_xgb.update({"n_estimators": 300, "random_state": 42, "use_label_encoder": False, "eval_metric": "logloss"})
print("Mejores hiperparámetros XGBoost:", best_params_xgb)

# === 5. MODELOS DEFINITIVOS ===

lgb = LGBMClassifier(**best_params_lgb)
xgb = XGBClassifier(**best_params_xgb)

# Entrenamiento de ambos modelos para stacking
lgb.fit(X_scaled, y)
xgb.fit(X_scaled, y)

joblib.dump(lgb, "lgb_model.joblib")
joblib.dump(xgb, "xgb_model.joblib")

# === 6. STACKING ENSEMBLE ===

stack = StackingClassifier(
    estimators=[
        ("lgb", lgb),
        ("xgb", xgb),
    ],
    final_estimator=LGBMClassifier(random_state=42),
    passthrough=True, n_jobs=-1
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(stack, X_scaled, y, cv=cv, scoring="accuracy")
print(f"🏆 Accuracy Stacking (CV): {scores.mean():.4f} ± {scores.std():.4f}")

stack.fit(X_scaled, y)
joblib.dump(stack, "stacking_model.joblib")

# === 7. PREDICCIÓN Y SUBMISSION ===

pred = stack.predict(X_test_scaled)
submission = pd.read_csv("gender_submission.csv")  # O crea tu propio DataFrame con PassengerId
submission["Survived"] = pred
submission.to_csv("submission.csv", index=False)
print("✅ submission.csv generado con éxito. ¡Listo para Kaggle!")
print("✅ Modelos y scaler guardados para reproducción y futuro uso.")

