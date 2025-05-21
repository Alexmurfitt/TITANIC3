import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectFromModel
import optuna
import shap
import warnings
warnings.filterwarnings('ignore')

# 1. Carga de datos
train = pd.read_csv('train_final_imputed_with_target.csv')
test = pd.read_csv('test_final_imputed.csv')

# 2. Separar X e y
y = train['Survived']
X = train.drop(['Survived'], axis=1)

# 3. Eliminación de features basura
def feature_selection(X, y, threshold='median'):
    lgbm_fs = LGBMClassifier(n_estimators=300, random_state=42)
    lgbm_fs.fit(X, y)
    selector = SelectFromModel(lgbm_fs, threshold=threshold, prefit=True)
    X_sel = selector.transform(X)
    cols_sel = X.columns[selector.get_support()]
    return X_sel, cols_sel, selector

X_sel, cols_sel, selector = feature_selection(X, y)
X = pd.DataFrame(X_sel, columns=cols_sel)
test = pd.DataFrame(selector.transform(test), columns=cols_sel)

# 4. Hiperparámetros LightGBM vía Optuna
def tune_lgbm(X, y):
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 20, 120),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
            'random_state': 42,
            'n_estimators': 300
        }
        model = LGBMClassifier(**params)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, valid_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            scores.append(accuracy_score(y_val, preds))
        return np.mean(scores)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25)
    return study.best_params

best_params_lgbm = tune_lgbm(X, y)
print('Mejores hiperparámetros LightGBM:', best_params_lgbm)

# 5. Modelos base
models = [
    ('lgbm', LGBMClassifier(**best_params_lgbm)),
    ('xgb', XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)),
    ('cat', CatBoostClassifier(verbose=0, random_state=42)),
    ('lr', LogisticRegression(max_iter=1500, random_state=42)),
]

# 6. Stacking-blending profesional
def stacking_blending(models, X, y, test, n_folds=5):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    S_train = np.zeros((X.shape[0], len(models)))
    S_test = np.zeros((test.shape[0], len(models)))
    for i, (name, model) in enumerate(models):
        S_test_i = np.zeros((test.shape[0], n_folds))
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            model.fit(X_tr, y_tr)
            S_train[val_idx, i] = model.predict(X_val)
            S_test_i[:, fold] = model.predict(test)
        S_test[:, i] = S_test_i.mean(axis=1)
    # Meta-modelo final
    meta_model = LogisticRegression(max_iter=1500, random_state=42)
    meta_model.fit(S_train, y)
    final_pred = meta_model.predict(S_test)
    return final_pred, S_train, S_test

final_pred, S_train, S_test = stacking_blending(models, X, y, test, n_folds=5)

# 7. SHAP interpretabilidad rápida
lgbm_model = LGBMClassifier(**best_params_lgbm)
lgbm_model.fit(X, y)
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar", show=False)

# 8. Guardar submission
submission = pd.read_csv('test.csv')
submission['Survived'] = final_pred.astype(int)
submission[['PassengerId', 'Survived']].to_csv('submission_sota.csv', index=False)
print('✅ Submission SOTA generado: submission_sota.csv')
