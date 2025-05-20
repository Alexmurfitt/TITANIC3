# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer

# 1. Cargar datos
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Guardar PassengerId de test para submission final
test_passenger_ids = test['PassengerId']

# Guardar variable objetivo y eliminarla del train para concatenar
y = train['Survived']
train.drop(['Survived'], axis=1, inplace=True)

# Marcar origen para luego separar
train['is_train'] = 1
test['is_train'] = 0

data = pd.concat([train, test], sort=False, ignore_index=True)

# 2. Imputación de valores nulos

# Cabin: Demasiados nulos, convertir a binaria 'Has_Cabin'
data['Has_Cabin'] = data['Cabin'].notnull().astype(int)

# Embarked: Imputar con la moda
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Fare: Solo 1 nulo en test, imputar con la mediana del Pclass correspondiente
data['Fare'] = data.groupby('Pclass')['Fare'].transform(
    lambda x: x.fillna(x.median())
)

# Age: Imputar con mediana por título extraído
def get_title(name):
    import re
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""
data['Title'] = data['Name'].apply(get_title)

# Simplificar títulos raros
rare_titles = ['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 
               'Sir', 'Jonkheer', 'Dona']
data['Title'] = data['Title'].replace(rare_titles, 'Rare')
data['Title'] = data['Title'].replace(['Mlle', 'Ms'], 'Miss')
data['Title'] = data['Title'].replace('Mme', 'Mrs')

# Imputar Age con la mediana por Title
data['Age'] = data.groupby('Title')['Age'].transform(lambda x: x.fillna(x.median()))

# 3. Feature engineering

# Family size
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = (data['FamilySize'] == 1).astype(int)

# Categorizar edad
data['AgeBin'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 60, 100], labels=False)

# Categorizar tarifa
data['FareBin'] = pd.qcut(data['Fare'], 4, labels=False)

# 4. Codificación de variables categóricas

# Label encoding para Sex, Embarked y Title
for col in ['Sex', 'Embarked', 'Title']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# 5. Eliminar columnas irrelevantes
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data.drop(columns=drop_cols, inplace=True)

# 6. Escalado de variables numéricas
num_cols = ['Age', 'Fare', 'FamilySize']
scaler = RobustScaler()
data[num_cols] = scaler.fit_transform(data[num_cols])

# 7. Separar train y test procesados
train_prep = data[data['is_train']==1].drop(['is_train'], axis=1).copy()
test_prep = data[data['is_train']==0].drop(['is_train'], axis=1).copy()

# Añadir de nuevo la variable objetivo a train
train_prep['Survived'] = y.values

# 8. Guardar datasets limpios
train_prep.to_csv('train_clean.csv', index=False)
test_prep.to_csv('test_clean.csv', index=False)
pd.DataFrame({'PassengerId': test_passenger_ids}).to_csv('test_ids.csv', index=False)

print("✅ Preprocesamiento completado. Datasets exportados como train_clean.csv y test_clean.csv")
