import pandas as pd
import joblib
import os

# 1. Cargar el test y la estructura de columnas del train
test_df = pd.read_csv('test.csv')
cols_train = pd.read_csv('train_final_imputed.csv').columns

# 2. Feature engineering: cargar matrices combinadas
df_all = pd.read_csv('feature_engineered_all.csv')
feature_matrix = pd.read_csv('feature_matrix_all.csv')

# Selección de filas de test (asume columna 'set' == 'test')
test_idx = df_all['set'] == 'test'
X_test_raw = feature_matrix[test_idx]

# 3. OneHot encoding idéntico al train
X_test_encoded = pd.get_dummies(X_test_raw, drop_first=True)
X_test_encoded = X_test_encoded.reindex(columns=cols_train, fill_value=0)

# 4. Imputación con el imputer serializado, si existe
imputer_path = 'models/imputer.joblib'
if os.path.exists(imputer_path):
    imputer = joblib.load(imputer_path)
    X_test_final = pd.DataFrame(imputer.transform(X_test_encoded), columns=X_test_encoded.columns, index=X_test_encoded.index)
else:
    if X_test_encoded.isnull().sum().sum() > 0:
        print("⚠️ ¡Ojo! El test aún tiene nulos y no hay imputer guardado.")
    X_test_final = X_test_encoded

# 5. Exportar
X_test_final.to_csv('test_final_imputed.csv', index=False)
print("✅ Test procesado idéntico a train y guardado en test_final_imputed.csv")
