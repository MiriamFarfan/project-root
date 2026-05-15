import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(config: dict):
    # Carga del dataset
    raw_path = config["paths"]["raw"]
    df = pd.read_csv(raw_path)
    print(f"[data_loader] Dataset cargado: {df.shape[0]} filas y {df.shape[1]} columnas.")

    # Limpia TotalCharges
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_nulls = df["TotalCharges"].isna().sum()
    if n_nulls > 0:
        median_val = df["TotalCharges"].median()
        df["TotalCharges"] = df["TotalCharges"].fillna(median_val)
        print(f"[data_loader] TotalCharges: {n_nulls} nulos imputados con mediana ({median_val:.2f})")

    # Eliminación de columna irrelevante
    df = df.drop(columns=["customerID"])
    
    # Codificación de variables binarias 
    binary_map = {"Yes": 1, "No": 0}
    gender_map = {"Male": 1, "Female": 0}
 
    df["gender"]  = df["gender"].map(gender_map)
    df["Partner"] = df["Partner"].map(binary_map)
    df["Churn"]   = df["Churn"].map(binary_map)
 
    print("[data_loader] Codificación binaria aplicada: gender, Partner, Churn")
 
    # Separación de features(X) y target(y)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    
    # Train/Test split con parametros del config
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    print(f"[data_loader] Split: train={X_train.shape[0]} filas | test={X_test.shape[0]} filas")
    return X_train, X_test, y_train, y_test

