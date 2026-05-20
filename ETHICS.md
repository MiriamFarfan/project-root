# Análisis Ético y de Sesgos

## Contexto
El proyecto utiliza un modelo de Machine Learning para predecir el abandono de clientes (Churn) en telecomunicaciones.

Aunque el objetivo es mejorar la retención de clientes, existen implicaciones éticas relacionadas con privacidad, sesgos y uso responsable de los datos.

---

## Posibles Sesgos del Dataset

### 1. Sesgo Demográfico
Variables como:
- gender
- SeniorCitizen
- Partner

pueden introducir sesgos en el modelo si ciertos grupos están sobrerrepresentados.

---

### 2. Sesgo Económico
Variables relacionadas con:
- MonthlyCharges
- Contract
- PaymentMethod

pueden afectar de manera desigual a clientes con menor capacidad económica.

---

### 3. Representatividad
El dataset proviene de una única empresa de telecomunicaciones, por lo que el modelo podría no generalizar correctamente a otros contextos o países.

---

## Riesgos Éticos

### Discriminación Automatizada
El modelo podría generar decisiones injustas hacia ciertos perfiles de clientes.

### Privacidad
El manejo de información de clientes debe cumplir principios de protección de datos y confidencialidad.

### Dependencia del Modelo
Las decisiones empresariales no deben depender únicamente de predicciones automáticas.

---

## Medidas de Mitigación

- Eliminación de identificadores directos (`customerID`).
- Evaluación continua del desempeño del modelo.
- Revisión humana en decisiones críticas.
- Monitoreo de sesgos y métricas de equidad.

---

## Conclusión
El uso de Machine Learning en problemas de churn puede aportar beneficios operativos importantes, pero debe implementarse con supervisión humana, transparencia y evaluación ética continua.