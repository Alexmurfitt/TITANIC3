import os

# 1. ¿Existe un modelo AutoGluon?
autogluon_path = "AutogluonModels/ag-20250521_102821"
lgbm_path = "models/lgbm_best_model.pkl"

if os.path.exists(autogluon_path):
    # --- AUTOGLUON PIPELINE ---
    print("🟢 Usando modelo AutoGluon para máxima eficiencia y precisión.")
    from autogluon.tabular import TabularPredictor
    import pandas as pd
    predictor = TabularPredictor.load(autogluon_path)
    test = pd.read_csv("test.csv")
    preds = predictor.predict(test)
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": preds.astype(int)
    })
    submission.to_csv("submission.csv", index=False)
    print("✅ Submission generado por AutoGluon: submission.csv")

elif os.path.exists(lgbm_path):
    # --- PIPELINE CLÁSICO ---
    print("🟡 Usando modelo LGBM clásico, ejecutando pipeline_test_apply + submission.py...")
    import pandas as pd
    import joblib

    # Paso 1: pipeline_test_apply.py
    test_df = pd.read_csv('test.csv')
    cols_train = pd.read_csv('train_final_imputed.csv').columns
    df_all = pd.read_csv('feature_engineered_all.csv')
    feature_matrix = pd.read_csv('feature_matrix_all.csv')
    test_idx = df_all['set'] == 'test'
    X_test_raw = feature_matrix[test_idx]
    X_test_encoded = pd.get_dummies(X_test_raw, drop_first=True)
    X_test_encoded = X_test_encoded.reindex(columns=cols_train, fill_value=0)
    imputer_path = 'models/imputer.joblib'
    if os.path.exists(imputer_path):
        imputer = joblib.load(imputer_path)
        X_test_final = pd.DataFrame(imputer.transform(X_test_encoded), columns=X_test_encoded.columns, index=X_test_encoded.index)
    else:
        X_test_final = X_test_encoded
    X_test_final.to_csv('test_final_imputed.csv', index=False)

    # Paso 2: submission.py
    model = joblib.load(lgbm_path)
    preds = model.predict(X_test_final)
    submission = pd.DataFrame({'PassengerId': test_df['PassengerId'], 'Survived': preds.astype(int)})
    submission.to_csv('submission.csv', index=False)
    print('✅ Submission generado con modelo clásico: submission.csv')

else:
    print("❌ No se encontró un modelo entrenado válido. Revisa la carpeta models/ y AutogluonModels/.")

