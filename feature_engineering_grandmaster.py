# feature_engineering_grandmaster.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
import featuretools as ft
import joblib

# ==============================
# 1. CARGA Y UNIÓN DE DATOS
# ==============================
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train['set'] = 'train'
test['set'] = 'test'
df_all = pd.concat([train, test], sort=False).reset_index(drop=True)

# ==============================
# 2. FEATURE ENGINEERING MANUAL SOTA
# ==============================

# Título del pasajero
df_all['Title'] = df_all['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
df_all['Title'] = df_all['Title'].replace({
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs", "Lady": "Rare", "Countess": "Rare",
    "Capt": "Rare", "Col": "Rare", "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare",
    "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare"
})

# Tamaño familiar y si está solo
df_all['FamilySize'] = df_all['SibSp'] + df_all['Parch'] + 1
df_all['IsAlone'] = (df_all['FamilySize'] == 1).astype(int)

# Deck (cubierta) desde la cabina
df_all['Deck'] = df_all['Cabin'].str[0].fillna('U')

# Agrupación de tickets
ticket_counts = df_all['Ticket'].value_counts()
df_all['TicketGroup'] = df_all['Ticket'].map(ticket_counts)
df_all['TicketGroup'] = pd.cut(df_all['TicketGroup'], bins=[0, 1, 4, 8, 1000], labels=['Solo', 'Small', 'Medium', 'Large'])

# Binning de edad y tarifa (KBinsDiscretizer robusto)
age_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
fare_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
df_all['AgeBin'] = age_bins.fit_transform(df_all[['Age']].fillna(-0.5)).astype(int)
df_all['FareBin'] = fare_bins.fit_transform(df_all[['Fare']].fillna(-0.5)).astype(int)

# Variable RescuePriority histórica
def rescue_priority(row):
    if row['Sex'] == 'female' or (not pd.isnull(row['Age']) and row['Age'] < 15):
        return 3
    elif row['Sex'] == 'male' and row['Pclass'] == 1:
        return 2
    else:
        return 1

df_all['RescuePriority'] = df_all.apply(rescue_priority, axis=1)

# Interacciones clave
df_all['Sex_Pclass'] = df_all['Sex'].astype(str) + "_" + df_all['Pclass'].astype(str)
df_all['Embarked_FareBin'] = df_all['Embarked'].astype(str) + "_" + df_all['FareBin'].astype(str)
df_all['Title_Pclass'] = df_all['Title'].astype(str) + "_" + df_all['Pclass'].astype(str)

# Family Survival Rate (solo con los datos de train, sin leakage)
df_all_train = df_all[df_all['set'] == 'train']
if 'Survived' in df_all_train:
    family_survival = df_all_train.groupby(['FamilySize', 'Pclass'])['Survived'].mean().to_dict()
    df_all['Family_Survival_Rate'] = df_all.apply(
        lambda x: family_survival.get((x['FamilySize'], x['Pclass']), np.nan), axis=1)
else:
    df_all['Family_Survival_Rate'] = np.nan

# Guarda el CSV con features manuales (opcional, para trazabilidad)
df_all.to_csv('feature_engineered_all_grandmaster.csv', index=False)

print("✅ Feature engineering manual SOTA completado y guardado en feature_engineered_all_grandmaster.csv")

# ==============================
# 3. FEATURETOOLS (DFS AUTOMÁTICO)
# ==============================
print("⏳ Generando features automáticas con Featuretools (DFS)...")
es = ft.EntitySet(id='titanic')
es = es.add_dataframe(dataframe_name='passengers', dataframe=df_all, index='PassengerId')

feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='passengers',  # NUEVA API Featuretools
    agg_primitives=[
        "mean", "sum", "count", "max", "min", "std", "mode", "num_unique"
    ],
    trans_primitives=[
        "year", "month", "day", "weekday", "is_weekend", "not", "absolute"
    ],
    max_depth=2,
    verbose=True
)

# Limpieza: Reset index y nombre para evitar duplicados al guardar
feature_matrix.reset_index(drop=True, inplace=True)
feature_matrix.index.name = None

# ==============================
# 4. GUARDADO FINAL
# ==============================
feature_matrix.to_csv('feature_matrix_all_grandmaster.csv', index=False)
print("✅ Feature matrix completa generada y guardada en feature_matrix_all_grandmaster.csv")

# Guarda los definitions (opcional, para reproducibilidad total)
joblib.dump(feature_defs, 'featuretools_feature_defs_grandmaster.joblib')
print("✅ Definitions guardados en featuretools_feature_defs_grandmaster.joblib")
