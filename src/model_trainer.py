# src/model_trainer.py

import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score


def train_and_save_model(X_train, y_train, X_test, y_test, config):
    """
    Entrena un modelo de clasificación y lo guarda en disco.

    Parámetros:
        X_train, y_train: datos de entrenamiento
        X_test, y_test:   datos de prueba
        config:           diccionario cargado desde params.yaml

    Retorna:
        dict con keys: accuracy, recall, f1_score
    """

    # --- 1. Fábrica de modelos: elige según config ---
    model_name = config["model"]["name"]

    if model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=config["model"].get("n_estimators", 100),
            max_depth=config["model"].get("max_depth", None),
            random_state=config["data"].get("random_state", 42)
        )

    elif model_name == "LogisticRegression":
        model = LogisticRegression(
            C=config["model"].get("C", 1.0),
            max_iter=config["model"].get("max_iter", 1000),
            random_state=config["data"].get("random_state", 42)
        )

    elif model_name == "SVM":
        model = SVC(
            C=config["model"].get("C", 1.0),
            kernel=config["model"].get("kernel", "rbf"),
            random_state=config["data"].get("random_state", 42)
        )

    else:
        raise ValueError(f"Modelo no reconocido: '{model_name}'. "
                         f"Opciones válidas: RandomForest, LogisticRegression, SVM")

    # --- 2. Entrenamiento ---
    model.fit(X_train, y_train)

    # --- 3. Predicción y métricas ---
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  accuracy_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1_score":  f1_score(y_test, y_pred)
    }

    print(f"Modelo: {model_name}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")

    # --- 4. Guardar modelo ---
    save_path = config["paths"]["model_save"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Modelo guardado en: {save_path}")

    return metrics