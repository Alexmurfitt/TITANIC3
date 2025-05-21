import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

df_all = pd.read_csv('feature_engineered_all.csv')
feature_matrix = pd.read_csv('feature_matrix_all.csv')

# Target solo para train
target = df_all.loc[df_all['set'] == 'train', 'Survived']
X = feature_matrix.iloc[:len(target)]

# Convierte todas las columnas categóricas a variables numéricas (OneHot)
X_encoded = pd.get_dummies(X, drop_first=True)

# Reemplaza NaN, inf y -inf por valores finitos
X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan).fillna(-999)

rf = RandomForestClassifier(random_state=42)
rf.fit(X_encoded, target)

importances = pd.Series(rf.feature_importances_, index=X_encoded.columns).sort_values(ascending=False)
print("\nTOP 15 FEATURES MÁS IMPORTANTES:")
print(importances.head(15))
