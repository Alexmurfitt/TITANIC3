# pipeline_grandmaster.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import shap
import lightgbm as lgb
from autogluon.tabular import TabularPredictor
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 1. Carga datos avanzados (features manuales + DFS)
print("🟢 Cargando features avanzadas...")
df_all = pd.read_csv("feature_matrix_all_grandmaster.csv")
if 'set' not in df_all.columns:
    df_all['set'] = pd.read_csv("feature_engineered_all_grandmaster.csv")['set']

train_idx = df_all['set'] == 'train'
test_idx = df_all['set'] == 'test'

if 'Survived' not in df_all.columns:
    df_all['Survived'] = pd.read_csv('train.csv')['Survived'].tolist() + [np.nan]*sum(test_idx)

cols_to_drop = ['set', 'Survived']
if 'PassengerId' in df_all.columns:
    cols_to_drop.append('PassengerId')

X_train = df_all.loc[train_idx].drop(cols_to_drop, axis=1)
y_train = df_all.loc[train_idx]['Survived'].astype(int)
X_test = df_all.loc[test_idx].drop(cols_to_drop, axis=1)
passenger_ids_test = pd.read_csv("test.csv")["PassengerId"]

# 2. Imputación avanzada (KNNImputer para numéricas, SimpleImputer para categóricas)
print("🟢 Imputando nulos...")
num_cols = X_train.select_dtypes(include=[np.number]).columns
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns

# Imputers
knn_imputer = KNNImputer(n_neighbors=5)
simple_imputer = SimpleImputer(strategy='most_frequent')

X_train_num = pd.DataFrame(knn_imputer.fit_transform(X_train[num_cols]), columns=num_cols)
X_test_num = pd.DataFrame(knn_imputer.transform(X_test[num_cols]), columns=num_cols)

X_train_cat = pd.DataFrame(simple_imputer.fit_transform(X_train[cat_cols]), columns=cat_cols)
X_test_cat = pd.DataFrame(simple_imputer.transform(X_test[cat_cols]), columns=cat_cols)

# Save imputers
os.makedirs("models", exist_ok=True)
joblib.dump(knn_imputer, "models/knn_imputer.joblib")
joblib.dump(simple_imputer, "models/simple_imputer.joblib")

# 3. Encoding categórico (OneHotEncoder robusto)
print("🟢 Codificando variables categóricas...")
from sklearn import __version__ as skl_version
if int(skl_version.split(".")[1]) >= 2:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
else:
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

X_train_cat_enc = encoder.fit_transform(X_train_cat)
X_test_cat_enc = encoder.transform(X_test_cat)
joblib.dump(encoder, "models/ohe_encoder.joblib")

cat_feat_names = encoder.get_feature_names_out(cat_cols)
X_train_enc = np.hstack([X_train_num, X_train_cat_enc])
X_test_enc = np.hstack([X_test_num, X_test_cat_enc])
all_features = list(X_train_num.columns) + list(cat_feat_names)

# 4. Escalado
print("🟢 Escalando features...")
scaler = StandardScaler()
X_train_enc = scaler.fit_transform(X_train_enc)
X_test_enc = scaler.transform(X_test_enc)
joblib.dump(scaler, "models/scaler.joblib")

# 5. Selección de variables (Random Forest importance)
print("🟢 Seleccionando variables importantes (RF)...")
rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X_train_enc, y_train)
importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:60]  # top 60 features

X_train_sel = X_train_enc[:, top_idx]
X_test_sel = X_test_enc[:, top_idx]
sel_features = [all_features[i] for i in top_idx]

# 6. Modelado avanzado: LightGBM + CV manual con early stopping
print("🟢 Modelando con LightGBM + StratifiedKFold + EarlyStopping...")
params = {
    'objective': 'binary',
    'metric': 'binary_error',   # binary_error = 1-accuracy
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42
}
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_iters = []
scores = []

for fold, (tr_idx, val_idx) in enumerate(folds.split(X_train_sel, y_train)):
    print(f"\n--- Fold {fold+1} ---")
    X_tr, X_val = X_train_sel[tr_idx], X_train_sel[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(period=100)
        ]
    )
    best_iters.append(model.best_iteration)
    preds = model.predict(X_val, num_iteration=model.best_iteration)
    preds_bin = (preds > 0.5).astype(int)
    acc = accuracy_score(y_val, preds_bin)
    scores.append(acc)
    print(f"Fold {fold+1} | Best Iter: {model.best_iteration} | Val Accuracy: {acc:.4f}")


mean_iter = int(np.mean(best_iters))
print(f"\nMedia de best_iteration en CV: {mean_iter}")
print(f"Media de accuracy en CV: {np.mean(scores):.4f}")

# Entrena modelo final en todo el train
final_model = lgb.train(params, lgb.Dataset(X_train_sel, label=y_train), num_boost_round=mean_iter)
joblib.dump(final_model, "models/lgbm_best_model_grandmaster.pkl")

# 7. SHAP para interpretabilidad (solo primeras 500 muestras por performance)
print("🟢 Interpretabilidad con SHAP...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_train_sel[:500])
os.makedirs("plots", exist_ok=True)
shap.summary_plot(shap_values, features=X_train_sel[:500], feature_names=sel_features, show=False)
plt.savefig("plots/shap_summary_grandmaster.png")
print("✅ Gráfico SHAP guardado en plots/shap_summary_grandmaster.png")

# 8. AutoML benchmark (AutoGluon)
print("🟢 Benchmarking con AutoGluon (puede tardar)...")
train_autogluon = df_all.loc[train_idx, :].copy()
test_autogluon = df_all.loc[test_idx, :].copy()
train_autogluon['Survived'] = y_train
predictor = TabularPredictor(label='Survived', path='AutogluonModels_grandmaster').fit(
    train_autogluon.drop(columns=['set']), time_limit=3600, presets="best_quality")
preds_auto = predictor.predict(test_autogluon.drop(columns=['set', 'Survived']))
pd.DataFrame({'PassengerId': passenger_ids_test, 'Survived': preds_auto.astype(int)}).to_csv('submission_autogluon.csv', index=False)

# 9. Stacking/Blending avanzado (puedes añadir tu script aquí)
# -- Añade aquí tu ensemble si lo deseas --

# 10. Generación de submission final
print("🟢 Generando submission final LightGBM...")
preds_lgbm = final_model.predict(X_test_sel)
preds_lgbm = (preds_lgbm > 0.5).astype(int)
submission = pd.DataFrame({'PassengerId': passenger_ids_test, 'Survived': preds_lgbm})
submission.to_csv("submission_lgbm_grandmaster.csv", index=False)
print("✅ Submission LightGBM guardado como submission_lgbm_grandmaster.csv")

print("🟢 Pipeline Grandmaster completado. ¡Listo para entregar y defender!")
