import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

df_all = pd.read_csv('feature_engineered_all.csv')
feature_matrix = pd.read_csv('feature_matrix_all.csv')

train_idx = df_all['set'] == 'train'
X = feature_matrix[train_idx]
target = df_all.loc[train_idx, 'Survived']

# --- ENCODING (OneHot para todo) ---
X_encoded = pd.get_dummies(X, drop_first=True)

# 🔴 Nueva línea: limpia los infinitos y -inf
X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)

# --- IMPUTACIÓN AVANZADA ---
imputer = KNNImputer(n_neighbors=5)
X_imputed = pd.DataFrame(imputer.fit_transform(X_encoded), columns=X_encoded.columns, index=X_encoded.index)

X_imputed.to_csv('train_final_imputed.csv', index=False)
print("✅ Imputación avanzada y encoding completados. Dataset guardado como train_final_imputed.csv")

