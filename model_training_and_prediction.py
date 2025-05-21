import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna

# Cargar los datos generados antes
X = pd.read_csv("train_features.csv")
X_test = pd.read_csv("test_features.csv")
y = pd.read_csv("y.csv", header=None).iloc[:,0].values.astype(int)
test_ids = pd.read_csv("test_ids.csv")["PassengerId"]

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Selección de features automática
selector = SelectKBest(mutual_info_classif, k=30) # puedes ajustar k tras ver la importancia
X_sel = selector.fit_transform(X_scaled, y)
X_test_sel = selector.transform(X_test_scaled)

# Optuna para XGBoost (solo como ejemplo, puedes extender a otros)
def objective(trial):
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
        'eval_metric': 'logloss',
        'use_label_encoder': False
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X_sel, y):
        clf = xgb.XGBClassifier(**param, random_state=42)
        clf.fit(X_sel[train_idx], y[train_idx])
        pred = clf.predict(X_sel[val_idx])
        scores.append(accuracy_score(y[val_idx], pred))
    return np.mean(scores)

# Solo la primera vez: descomenta para optimizar XGBoost y usar los mejores hiperparámetros
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)
# print("Mejores hiperparámetros:", study.best_params)

# Hiperparámetros óptimos (ajusta tras optimización)
xgb_best_params = {
    'max_depth': 5,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'logloss',
    'use_label_encoder': False,
    'random_state': 42
}

lgbm_clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42)
cat_clf = cb.CatBoostClassifier(iterations=500, learning_rate=0.05, verbose=0, random_state=42)
xgb_clf = xgb.XGBClassifier(**xgb_best_params)

# Ensemble avanzado (Stacking)
stack = StackingClassifier(
    estimators=[
        ('xgb', xgb_clf),
        ('lgb', lgbm_clf),
        ('cat', cat_clf)
    ],
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=-1
)

# Validación cruzada y entrenamiento final
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = []
for train_idx, val_idx in skf.split(X_sel, y):
    stack.fit(X_sel[train_idx], y[train_idx])
    preds = stack.predict(X_sel[val_idx])
    scores.append(accuracy_score(y[val_idx], preds))
print(f"CV Accuracy media: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

# Entrena en todo el dataset
stack.fit(X_sel, y)

# Predicción final para test.csv
test_preds = stack.predict(X_test_sel)
output = pd.DataFrame({"PassengerId": test_ids, "Survived": test_preds.astype(int)})
output.to_csv("submission.csv", index=False)
print("✅ Predicción lista: submission.csv (envíala a Kaggle)")
