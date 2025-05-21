# titanic_autogluon.py
import pandas as pd
from autogluon.tabular import TabularPredictor

# Carga datos
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Autogluon requiere que la columna objetivo esté en el set de entrenamiento
label = 'Survived'

# Opción: puedes hacer ingeniería de features manual, pero Autogluon lo hace automáticamente.
# Solo elimina columnas que no aportan nada o son IDs.
drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']  # puedes dejar 'Cabin' si quieres, Autogluon la trata como categórica

X_train = train.drop(columns=drop_cols)
X_test = test.drop(columns=drop_cols)

# Si faltan valores, Autogluon los rellena automáticamente.

# Crea predictor y entrena
predictor = TabularPredictor(label=label, eval_metric='accuracy').fit(
    X_train,
    time_limit=1200,        # Máximo tiempo de entrenamiento (segundos)
    presets="best_quality", # Para sacar el máximo posible, aunque tarda más
    num_bag_folds=10,       # Ensembles fuertes (más robusto, tarda más)
    num_stack_levels=2      # Stacking más profundo (máxima precisión)
)

# Predicción
preds = predictor.predict(X_test)

# Exporta CSV listo para Kaggle
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': preds.astype(int)
})
submission.to_csv('submission_autogluon.csv', index=False)
print('✅ Predicción Autogluon lista: submission_autogluon.csv')
