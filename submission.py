from autogluon.tabular import TabularPredictor
import pandas as pd

try:
    print("Cargando modelo AutoGluon...")
    predictor = TabularPredictor.load("AutogluonModels/ag-20250521_102821")

    print("Cargando test.csv...")
    test = pd.read_csv("test.csv")
    print(f"Test shape: {test.shape}")

    print("Realizando predicción...")
    preds = predictor.predict(test)
    print(f"Predicciones hechas. N = {len(preds)}")

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": preds.astype(int)
    })

    output_file = "submission.csv"
    submission.to_csv(output_file, index=False)
    print(f"✅ ¡Archivo {output_file} guardado correctamente!")

except Exception as e:
    print(f"ERROR: {e}")
