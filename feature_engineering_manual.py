import pandas as pd
import numpy as np

# Carga de datos
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Unir datasets para engineering igualado
train_df['set'] = 'train'
test_df['set'] = 'test'
df_all = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

# 1. Extraer Título del nombre
df_all['Title'] = df_all['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
rare_titles = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df_all['Title'] = df_all['Title'].replace(['Mlle', 'Ms'], 'Miss').replace('Mme', 'Mrs')
df_all['Title'] = df_all['Title'].replace(rare_titles, 'Rare')

# 2. Tamaño familiar y si viaja solo
df_all['FamilySize'] = df_all['SibSp'] + df_all['Parch'] + 1
df_all['IsAlone'] = (df_all['FamilySize'] == 1).astype(int)

# 3. Deck y cabina conocida
df_all['Cabin_known'] = df_all['Cabin'].notnull().astype(int)
df_all['Deck'] = df_all['Cabin'].str[0].fillna('Unknown')

# 4. Grupo de ticket
df_all['TicketGroup'] = df_all.groupby('Ticket')['Ticket'].transform('count')

# 5. Binning inteligente para Age y Fare
df_all['AgeBin'] = pd.cut(df_all['Age'], bins=[0, 12, 20, 40, 60, 80], labels=False)
df_all['FareBin'] = pd.qcut(df_all['Fare'].fillna(df_all['Fare'].median()), 4, labels=False)

# Guardar dataset con nuevas features
df_all.to_csv('feature_engineered_all.csv', index=False)
print('✅ Feature engineering manual COMPLETO. Archivo guardado: feature_engineered_all.csv')


