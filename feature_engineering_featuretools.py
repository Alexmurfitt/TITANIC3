import pandas as pd
import featuretools as ft

# Cargar dataset generado manualmente
df_all = pd.read_csv('feature_engineered_all.csv')

cols_for_dfs = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title',
    'FamilySize', 'IsAlone', 'Deck', 'TicketGroup', 'AgeBin', 'FareBin'
]
df_dfs = df_all[cols_for_dfs].copy()

# Crear columna de ID única para Featuretools
df_dfs['PassengerID_FT'] = range(len(df_dfs))

es = ft.EntitySet(id="titanic_data")
es = es.add_dataframe(dataframe_name="passengers", dataframe=df_dfs, index='PassengerID_FT')


feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="passengers",
    trans_primitives=['add_numeric', 'multiply_numeric', 'divide_numeric', 'cum_mean', 'cum_sum'],
    max_depth=2,
    verbose=True
)

feature_matrix.to_csv('feature_matrix_all.csv', index=False)
print('✅ Feature Engineering automática COMPLETO. Archivo guardado: feature_matrix_all.csv')

