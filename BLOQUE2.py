# ----------------------------------------
# 🚢 TITANIC SOTA PIPELINE (2025) - BLOQUE 2
# Feature Engineering Manual + RescuePriority + Deep Feature Synthesis (opcional)
# ----------------------------------------

import pandas as pd
import numpy as np

# --------- Feature Engineering Manual ---------
def feature_engineering_manual(df):
    # 1. Extraer Título del nombre
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # 2. Family Size e indicador de estar solo/a
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # 3. Deck (primer carácter de Cabin)
    df['Cabin'] = df['Cabin'].fillna('U')
    df['Deck'] = df['Cabin'].apply(lambda x: x[0])

    # 4. Ticket Group: cuántas personas comparten el mismo ticket
    df['TicketGroup'] = df.groupby('Ticket')['Ticket'].transform('count')
    df['TicketGroup'] = df['TicketGroup'].apply(lambda x: x if x <= 4 else 4)

    # 5. Binning y tratamiento de Fare y Age
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['FareBin'] = pd.qcut(df['Fare'], 4, labels=False)
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 120], labels=[0, 1, 2, 3])

    # 6. Embarque missing
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    # 7. Eliminar columnas irrelevantes (guardar PassengerId para el submission)
    df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True, errors='ignore')

    # 8. NUEVO: Prioridad de salvamento histórica
    def rescue_priority(row):
        if row['Sex'] == 'female' or (row['Age'] < 15):
            return 3
        elif row['Pclass'] == 1:
            return 2
        else:
            return 1
    df['RescuePriority'] = df.apply(rescue_priority, axis=1)

    return df

train_fe = feature_engineering_manual(train.copy())
test_fe  = feature_engineering_manual(test.copy())

print("Feature engineering manual + RescuePriority completado.")
display(train_fe.head())
display(test_fe.head())
