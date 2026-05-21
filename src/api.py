"""
api.py — MLOps Engineer (Isay Morales)
API REST para predicción de Churn usando el modelo entrenado.

Uso:
    uvicorn src.api:app --reload

Documentación automática:
    http://127.0.0.1:8000/docs
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------------------------------------------------
# Inicializar app
# -----------------------------------------------------------------------
app = FastAPI(
    title="Churn Prediction API",
    description="Predice si un cliente de telecomunicaciones abandonará el servicio.",
    version="1.0.0",
)

# -----------------------------------------------------------------------
# Esquema de entrada (datos crudos del cliente, igual que el CSV)
# -----------------------------------------------------------------------
class Cliente(BaseModel):
    gender: str             # "Male" | "Female"
    SeniorCitizen: int      # 0 | 1
    Partner: str            # "Yes" | "No"
    Dependents: str         # "Yes" | "No"
    tenure: int             # meses como cliente
    PhoneService: str       # "Yes" | "No"
    MultipleLines: str      # "Yes" | "No" | "No phone service"
    InternetService: str    # "DSL" | "Fiber optic" | "No"
    OnlineSecurity: str     # "Yes" | "No" | "No internet service"
    OnlineBackup: str       # "Yes" | "No" | "No internet service"
    DeviceProtection: str   # "Yes" | "No" | "No internet service"
    TechSupport: str        # "Yes" | "No" | "No internet service"
    StreamingTV: str        # "Yes" | "No" | "No internet service"
    StreamingMovies: str    # "Yes" | "No" | "No internet service"
    Contract: str           # "Month-to-month" | "One year" | "Two year"
    PaperlessBilling: str   # "Yes" | "No"
    PaymentMethod: str      # "Electronic check" | "Mailed check" | etc.
    MonthlyCharges: float
    TotalCharges: float

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.0,
                "TotalCharges": 1020.0
            }
        }

# -----------------------------------------------------------------------
# Cargar modelo al iniciar (una sola vez)
# -----------------------------------------------------------------------
MODEL_PATH = "models/model.pkl"
model = None

@app.on_event("startup")
def cargar_modelo():  # noqa: B006
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Modelo cargado desde: {MODEL_PATH}")
    else:
        print(f"ADVERTENCIA: No se encontró el modelo en {MODEL_PATH}. Ejecuta python -m src.main primero.")

# -----------------------------------------------------------------------
# Preprocesamiento (mismo que data_loader.py)
# -----------------------------------------------------------------------
def preprocesar(cliente: Cliente) -> pd.DataFrame:
    data = cliente.dict()

    # Codificar binarios
    data["gender"] = 1 if data["gender"] == "Male" else 0
    data["Partner"] = 1 if data["Partner"] == "Yes" else 0

    df = pd.DataFrame([data])

    # One-hot encoding igual que data_loader
    multiclass_cols = [
        "Dependents", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod"
    ]
    df = pd.get_dummies(df, columns=multiclass_cols, drop_first=False)

    # Alinear columnas con las que espera el modelo
    if hasattr(model, "feature_names_in_"):
        df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    return df

# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------
@app.get("/")
def raiz():
    return {"mensaje": "Churn Prediction API activa. Ve a /docs para probarla."}


@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": model is not None}


@app.post("/predict")
def predecir(cliente: Cliente):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Ejecuta 'python -m src.main' primero."
        )

    df = preprocesar(cliente)
    prediccion = int(model.predict(df)[0])
    probabilidad = None
    if hasattr(model, "predict_proba"):
        probabilidad = round(float(model.predict_proba(df)[0][1]) * 100, 2)

    return {
        "churn": prediccion,
        "resultado": "CHURN - El cliente probablemente abandona" if prediccion == 1 else "NO CHURN - El cliente probablemente se queda",
        "probabilidad_churn_pct": probabilidad,
    }


# -----------------------------------------------------------------------
# Permite correr con: python src/api.py  (puerto 5000, igual que el README)
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
