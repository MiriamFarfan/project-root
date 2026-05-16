"""
main.py — MLOps Engineer (Isay Morales)
Orquesta el pipeline completo: carga datos -> entrena modelo -> muestra metricas.

Uso:
    python -m src.main
"""

import os
import sys
import yaml

# Asegurar que la raiz del proyecto este en el path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.data_loader import load_and_preprocess_data
from src.trainer_model import train_and_save_model


def load_config(path="config/params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    print("=== Pipeline Churn MLOps ===\n")

    # 1. Leer configuracion
    config = load_config()
    print(f"Modelo seleccionado: {config['model']['name']}\n")

    # 2. Cargar y preprocesar datos (Data Engineer)
    print(">> Paso 1: Cargando y preprocesando datos...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

    # 3. Entrenar y guardar modelo (ML Engineer)
    print("\n>> Paso 2: Entrenando modelo...")
    metrics = train_and_save_model(X_train, y_train, X_test, y_test, config)

    # 4. Mostrar resultados
    print("\n=== Resultados ===")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1 Score : {metrics['f1_score']:.4f}")
    print("\nPipeline finalizado correctamente.")


if __name__ == "__main__":
    main()
