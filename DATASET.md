# Dataset: Telco Customer Churn

## Descripción General
El dataset utilizado corresponde al problema de predicción de abandono de clientes (Churn) en una empresa de telecomunicaciones.

El objetivo es predecir si un cliente abandonará el servicio utilizando variables demográficas, contractuales y de consumo.

## Fuente
- Kaggle: Telco Customer Churn
- URL: https://www.kaggle.com/blastchar/telco-customer-churn

## Número de Registros
- 7043 filas
- 21 columnas

## Variable Objetivo
- Churn
    - Yes = el cliente abandona
    - No = el cliente permanece

## Variables Relevantes
- gender
- SeniorCitizen
- Partner
- Dependents
- tenure
- Contract
- MonthlyCharges
- TotalCharges
- PaymentMethod

## Limpieza Realizada
Se aplicaron las siguientes transformaciones:

1. Conversión de `TotalCharges` a tipo numérico.
2. Imputación de valores nulos usando la mediana.
3. Eliminación de `customerID`.
4. Codificación binaria de variables categóricas.
5. One-Hot Encoding para variables multiclase.

## División del Dataset
- Train: 80%
- Test: 20%

## Tipo de Problema
Clasificación binaria supervisada.