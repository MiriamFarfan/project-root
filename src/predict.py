import os
import joblib
import pandas as pd

def predict_new_customer(model_path="models/model.pkl"):
    """
    Carga el modelo guardado y predice sobre un cliente de ejemplo.
    """
    # Manejo de error si el modelo no existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No se encontró el modelo en '{model_path}'. "
            "Ejecuta primero: python -m src.main"
        )

    model = joblib.load(model_path)
    print(f"Modelo cargado desde: {model_path}")

    # Cliente de ejemplo (ajustar columnas según las generadas por get_dummies)
    # Este diccionario debe coincidir con las features del modelo entrenado
    sample = {
        'SeniorCitizen': [0],
        'tenure': [12],
        'MonthlyCharges': [65.0],
        'TotalCharges': [780.0],
        'gender': [1],
        'Partner': [1],
        # Agrega aquí las columnas one-hot que genera el data_loader
        # Ejemplo: 'InternetService_Fiber optic': [1], etc.
    }

    df_sample = pd.DataFrame(sample)

    # Alinear columnas con las del modelo
    model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
    if model_features is not None:
        df_sample = df_sample.reindex(columns=model_features, fill_value=0)

    prediction = model.predict(df_sample)
    probability = model.predict_proba(df_sample) if hasattr(model, 'predict_proba') else None

    result = "CHURN (abandona)" if prediction[0] == 1 else "NO CHURN (se queda)"
    print(f"\nPredicción para cliente de ejemplo: {result}")
    if probability is not None:
        print(f"Probabilidad de Churn: {probability[0][1]:.2%}")

    return prediction[0]

if __name__ == "__main__":
    predict_new_customer()