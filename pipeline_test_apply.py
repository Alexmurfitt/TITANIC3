import pandas as pd
import joblib
import os  # ← Añade esto

# 1. Carga test.csv y feature_matrix_all.csv
test_df = pd.read_csv('test.csv')
cols_train = pd.read_csv('train_final_imputed.csv').columns

df_all = pd.read_csv('feature_engineered_all.csv')
feature_matrix = pd.read_csv('feature_matrix_all.csv')
test_idx = df_all['set'] == 'test'
X_test_raw = feature_matrix[test_idx]

# 2. Encoding idéntico (usa las columnas del train)
X_test_encoded = pd.get_dummies(X_test_raw, drop_first=True)
X_test_encoded = X_test_encoded.reindex(columns=cols_train, fill_value=0)

# 3. Imputación idéntica
imputer_path = 'models/imputer.joblib'
if os.path.exists(imputer_path):
    imputer = joblib.load(imputer_path)
    X_test_final = pd.DataFrame(imputer.transform(X_test_encoded), columns=X_test_encoded.columns, index=X_test_encoded.index)
else:
    # O, si no guardaste el imputer, simplemente omite este paso si tu test ya no tiene nulos.
    X_test_final = X_test_encoded  # Si tu test no tiene nulos tras FE/encoding

X_test_final.to_csv('test_final_imputed.csv', index=False)
print("✅ Test procesado idéntico a train y guardado en test_final_imputed.csv")
