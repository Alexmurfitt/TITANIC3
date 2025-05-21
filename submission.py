import pandas as pd
import joblib

# Cambia aquí el nombre si tu modelo final es lgb_model.pkl o lgb_model.joblib
model = joblib.load('models/lgbm_best_model.pkl')
X_test = pd.read_csv('test_final_imputed.csv')

# Predice
preds = model.predict(X_test)

# Carga PassengerId y exporta submission.csv
test_df = pd.read_csv('test.csv')
submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': preds})
submission.to_csv('submission.csv', index=False)
print('✅ Submission generado correctamente: submission.csv')
