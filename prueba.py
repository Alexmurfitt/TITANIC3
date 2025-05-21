import pandas as pd

# 1. Cargar archivo original (asegúrate de que esté en la misma carpeta)
df_train_original = pd.read_csv('train.csv')

# 2. Cargar archivo de features finales (el tuyo, sin 'Survived')
df_features = pd.read_csv('train_final_imputed.csv')

# 3. Asegúrate de que los DataFrames están alineados (IMPORTANTE)
# Normalmente ambos deberían estar en el mismo orden, si no, hay que alinear por 'PassengerId'

# 4. Añade la columna 'Survived' al DataFrame de features
df_features['Survived'] = df_train_original['Survived']

# 5. Guarda el nuevo archivo corregido
df_features.to_csv('train_final_imputed_with_target.csv', index=False)

print('✅ train_final_imputed_with_target.csv generado correctamente con columna Survived.')
