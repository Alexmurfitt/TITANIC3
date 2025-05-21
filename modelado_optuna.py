import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
import optuna

# 1. Carga los datos imputados
X = pd.read_csv('train_final_imputed.csv')
y = pd.read_csv('feature_engineered_all.csv').loc[lambda df: df['set'] == 'train', 'Survived']

# 2. Configura la validación cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Definición de la función objetivo de Optuna
def objective(trial):
    # Hiperparámetros de LightGBM (puedes cambiar a XGBoost o RF)
    param = {
        'objective': 'binary',
        'metric': 'accuracy',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 8, 128),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 5.0),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 5.0),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0)
    }
    # Entrenamiento y validación CV
    lgbm = lgb.LGBMClassifier(**param, n_estimators=250, random_state=42)
    scores = cross_val_score(lgbm, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean()

# 4. Ejecución de la optimización con Optuna
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print('✅ Mejor accuracy CV:', study.best_value)
print('✅ Mejores hiperparámetros:', study.best_params)

# 5. Entrena el modelo final con los mejores hiperparámetros y TODO el train
best_params = study.best_params.copy()
best_params.update({'objective': 'binary', 'metric': 'accuracy', 'boosting_type': 'gbdt', 'verbosity': -1})
model = lgb.LGBMClassifier(**best_params, n_estimators=250, random_state=42)
model.fit(X, y)

import joblib
import os
os.makedirs('models', exist_ok=True)  # IMPORTANTE: asegura que exista la carpeta

joblib.dump(model, 'models/lgbm_best_model.pkl')
print('✅ Modelo final entrenado y guardado como models/lgbm_best_model.pkl')
