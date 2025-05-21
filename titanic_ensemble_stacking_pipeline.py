# -*- coding: utf-8 -*-
"""
Pipeline avanzado Titanic ML: Stacking + Feature Selection + Calibración + SHAP
"""
import sys
print("Python ejecutando:", sys.executable)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from catboost import CatBoostClassifier
from borutashap import BorutaShap
import shap
import matplotlib.pyplot as plt

# Cargar tus datos preprocesados aquí
train = pd.read_csv('train_feature_engineered.csv')  # Usa el nombre de tu archivo procesado
test = pd.read_csv('test_feature_engineered.csv')

# Separar X e y (ajusta nombres de columnas según tu script)
X = train.drop(['Survived', 'PassengerId'], axis=1)
y = train['Survived']
X_test = test.drop(['PassengerId'], axis=1)
test_ids = test['PassengerId']

# 1. Separa el hold-out (nunca lo uses para tuning ni selección de features)
X_dev, X_holdout, y_dev, y_holdout = train_test_split(
    X, y, test_size=0.12, stratify=y, random_state=42
)

# 2. Selección de features avanzada con BorutaShap
# (Usa LightGBM o CatBoost como modelo base, solo sobre X_dev)
print("⏳ Ejecutando BorutaSHAP (puede tardar un poco)...")
feature_selector = BorutaShap(
    model=lgb.LGBMClassifier(n_jobs=-1),
    importance_measure='shap',
    classification=True
)
feature_selector.fit(X_dev, y_dev, n_trials=80, sample=False, train_or_test='test', normalize=True)
selected_features = feature_selector.Subset().columns.tolist()
print(f"✅ Features seleccionadas: {selected_features}")

# Opcional: Visualiza la importancia de las features
feature_selector.plot_features(importances_type='shap')

# Usa solo las features seleccionadas
X_dev_selected = X_dev[selected_features]
X_holdout_selected = X_holdout[selected_features]
X_test_selected = X_test[selected_features]

# 3. Define modelos base y meta-modelo
lgbm = lgb.LGBMClassifier(n_jobs=-1, random_state=42)  # Puedes poner tus hiperparámetros tuneados
cat = CatBoostClassifier(verbose=0, random_state=42)
meta_model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)

# 4. Stacking avanzado
stack = StackingClassifier(
    estimators=[
        ('lgbm', lgbm),
        ('cat', cat)
    ],
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1,
    passthrough=False
)
print("⏳ Entrenando Stacking...")
stack.fit(X_dev_selected, y_dev)
print("✅ Stacking entrenado")

# 5. Calibración con hold-out
print("⏳ Calibrando probabilidades con hold-out (Isotonic Regression)...")
calibrated_stack = CalibratedClassifierCV(stack, method='isotonic', cv='prefit')
calibrated_stack.fit(X_holdout_selected, y_holdout)
print("✅ Calibración terminada")

# 6. Predicción sobre el test
y_pred_proba = calibrated_stack.predict_proba(X_test_selected)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': y_pred})
submission.to_csv('submission.csv', index=False)
print("✅ Archivo 'submission.csv' generado")

# 7. Interpretabilidad: SHAP para el stacking
print("⏳ Calculando SHAP values...")
explainer = shap.Explainer(stack.named_estimators_['lgbm'], X_dev_selected)
shap_values = explainer(X_dev_selected)

# Summary plot de SHAP (importancia global de features)
shap.summary_plot(shap_values, X_dev_selected, plot_type='bar', show=True)
plt.savefig("shap_summary_plot.png")
print("✅ SHAP summary plot guardado como 'shap_summary_plot.png'")

# (Opcional) Muestra interpretabilidad de una predicción individual
idx = 0  # Cambia el índice para diferentes pasajeros
shap.plots.waterfall(shap_values[idx], show=True)

# 8. BONUS: Métrica en hold-out (para reportar tu score "honesto")
holdout_pred = calibrated_stack.predict(X_holdout_selected)
from sklearn.metrics import accuracy_score
holdout_acc = accuracy_score(y_holdout, holdout_pred)
print(f"🔎 Accuracy real en hold-out (sin data leakage): {holdout_acc:.5f}")
