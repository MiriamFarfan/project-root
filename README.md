📡 Proyecto Colaborativo MLOps: Predicción de Churn (Abandono de Clientes)
🎯 Objetivo del Proyecto
---
📂 El Dataset
Todos los equipos trabajarán con el dataset Telco Customer Churn.
Fuente: Kaggle - Telco Customer Churn
Archivo: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
Problema: Clasificación Binaria (¿El cliente se va? `Yes`/`No`)
Instrucción Importante:
Descarguen el CSV.
Guárdenlo en la carpeta `data/raw/`.
NO suban el CSV a Git (ya está configurado en `.gitignore`). Cada alumno debe descargarlo localmente.
> Para una descripción completa del dataset, sus aplicaciones y sesgos, consultar [`DATASET.md`](DATASET.md).
---
👥 Roles y Responsabilidades (Equipos de 4)
1. 👷 Data Engineer (`src/data_loader.py`)
Tu misión: Transformar datos brutos y sucios en datos limpios listos para entrenar.
Cargar el CSV desde `data/raw/`.
Limpieza: La columna `TotalCharges` tiene espacios vacíos `" "` en lugar de nulos. Convertirla a numérico y manejar los NaN resultantes.
Preprocesamiento: Eliminar `customerID`. Codificar variables binarias de Texto a 0/1.
División: Separar en Train/Test usando `test_size` y `random_state` de `config/params.yaml`.
Entregable: Función `load_and_preprocess_data(config)` que retorna `X_train, X_test, y_train, y_test`.
2. 🧠 ML Engineer (`src/model_trainer.py`)
Tu misión: Experimentar con algoritmos y guardar el mejor modelo.
Implementar una "Fábrica de Modelos" con al menos dos algoritmos (RandomForest, LogisticRegression y/o SVM).
Calcular métricas clave: Accuracy, Recall y F1-Score.
Guardar el modelo entrenado en `models/` usando `joblib`.
Entregable: Función `train_and_save_model(X_train, y_train, X_test, y_test, config)`.
3. ⚙️ MLOps Engineer (`src/main.py` y `config/`)
Tu misión: Orquestar el flujo y gestionar la configuración externa.
Crear y mantener `config/params.yaml` con parámetros de datos, modelo y rutas.
Escribir `src/main.py` que importe y ejecute en orden las funciones de los otros roles.
Asegurar que el proyecto corra con: `python -m src.main`.
Entregable: Un `main.py` funcional que lea el YAML y ejecute el pipeline completo.
4. 🛡️ QA & Production Engineer (`src/predict.py` y `tests/`)
Tu misión: Validar que el sistema funcione y preparar la inferencia para nuevos datos.
Crear `src/predict.py` que cargue `models/model.pkl` y prediga sobre un nuevo cliente.
Manejo de errores claro si el modelo no existe.
Escribir al menos 2 tests unitarios en `tests/test_pipeline.py`.
Entregable: Script de predicción robusto y tests pasando.
---
🚀 Flujo de Trabajo con Git
Clonar: `git clone <url-del-repo-del-equipo>`
Ramas: Cada alumno crea su rama:
```bash
   git checkout -b feature/data-engineer
   git checkout -b feature/ml-engineer
   git checkout -b feature/mlops-engineer
   git checkout -b feature/qa-engineer
   ```
Desarrollo: Trabajen en paralelo. Hagan commits frecuentes.
Integración: El MLOps Engineer crea el Pull Request integrando todas las ramas a `main`.
Prueba Final: `python -m src.main` en la rama `main`. Si corre, ¡misión cumplida!
---
📂 Estructura de Carpetas
```text
churn-mlops-project/
├── config/
│   └── params.yaml          # Configuración centralizada
├── data/
│   ├── raw/                 # WA_Fn-UseC_-Telco-Customer-Churn.csv (NO SUBIR)
│   └── processed/           # (Opcional) Datos limpios
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Rol: Data Engineer
│   ├── model_trainer.py     # Rol: ML Engineer
│   ├── main.py              # Rol: MLOps Engineer
│   ├── predict.py           # Rol: QA Engineer
│   └── api.py               # API REST (Flask)
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py     # Rol: QA Engineer
├── models/                  # Modelos .pkl generados (NO SUBIR o subir solo el final)
├── requirements.txt         # Dependencias con versiones específicas
├── .gitignore
├── DATASET.md               # Documentación del dataset
├── ETHICS.md                # Análisis ético y de sesgos
└── README.md                # Este archivo
```
---
⚙️ Instalación
1. Clonar el repositorio
```bash
git clone <url-del-repo-del-equipo>
cd churn-mlops-project
```
2. Crear y activar un entorno virtual
```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```
3. Instalar dependencias
```bash
pip install -r requirements.txt
```
El archivo `requirements.txt` incluye todas las librerías necesarias con versiones fijas para garantizar reproducibilidad:
```
pandas==2.2.2
scikit-learn==1.5.0
joblib==1.4.2
pyyaml==6.0.1
flask==3.0.3
pytest==8.2.2
numpy==1.26.4
```
---
▶️ Ejecución del Proyecto
Ejecutar el pipeline completo
```bash
python -m src.main
```
Hacer una predicción individual
```bash
python -m src.predict
```
Ejecutar pruebas unitarias
```bash
python -m pytest
```
---
🌐 API REST
Iniciar el servidor
```bash
python src/api.py
```
El servidor estará disponible en `http://localhost:5000`.
Endpoint disponible
Método	Ruta	Descripción
POST	`/predict`	Recibe datos de un cliente y devuelve la predicción de churn
Ejemplo con `curl`
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.2,
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check"
  }'
```
Ejemplo de respuesta
```json
{
  "churn": "No",
  "probability": 0.129
}
```
Ejemplo con Python `requests`
```python
import requests

data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "tenure": 12,
    "MonthlyCharges": 70.35,
    "TotalCharges": 844.2
}

response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())
```
---
📊 Resultados Obtenidos
Modelo seleccionado: RandomForest
Métrica	Resultado
Accuracy	0.8126
Recall	0.5362
F1 Score	0.6024
```
Predicción: NO CHURN
Probabilidad de Churn: 12.90%
```
---

✅ Checklist de Entrega
[x] El comando `python -m src.main` ejecuta todo el pipeline sin errores.
[x] El archivo `config/params.yaml` existe y controla los hiperparámetros.
[x] Hay al menos 2 modelos diferentes implementados en el código.
[x] El script `predict.py` carga el modelo y hace una predicción de ejemplo.
[x] El historial de Git muestra contribuciones de los 4 miembros del equipo.
[x] El `README.md` final incluye los resultados obtenidos (Accuracy/Recall del mejor modelo).
[x] El `requirements.txt` incluye todas las librerías con versiones específicas.
[x] Existe `DATASET.md` documentando el dataset y sus aplicaciones.
[x] Existe `ETHICS.md` con el análisis de sesgos y consideraciones éticas.

---
👤 Data Engineer - Miriam Jacquelin Becerra Farfán
Herramienta: ChatGPT/Claude
Uso ChatGPT: Usé ChatGPT para comprender el flujo de trabajo con Git (creación de ramas, commits y Pull Requests) y resolver dudas sobre cómo estructurar los archivos del proyecto.
Uso Claude: Generé la estructura de la función load_and_preprocess_data() y validé el manejo de valores nulos en TotalCharges.
👤 ML Engineer - José Ángel García Valle
-Me apoyé con Claude para crear la estructura base de la función train_and_save_model con el patrón de fábrica de modelos (if/elif) para seleccionar entre RandomForest, LogisticRegression y SVM. -También Claude ayudó a estructurar el cálculo de las 3 métricas: accuracy, recall y f1_score.

👤 QA Engineer - Carolina Ruiz Gudiño
Herramienta: Claude.
Uso de IA: Me ayudó Claude a generar el esqueleto de las pruebas unitarias con pytest para validar el pipeline.
👤 MLOps Engineer - Isay Morales
Herramienta: Claude
Uso de IA: Me apoyé en Claude para estructurar el config/params.yaml con las secciones correctas (data, model, paths) que necesitaban los módulos del Data Engineer y del ML Engineer. También me ayudó a orquestar el pipeline completo con un solo comando (python -m src.main).
---
📄 Documentación adicional
`DATASET.md` — Descripción del dataset, fuente, aplicaciones prácticas y sesgos observados.
`ETHICS.md` — Análisis de equidad, transparencia y limitaciones del modelo.
