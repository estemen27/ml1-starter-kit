# Competencia ML1: starter kit

**Problema:** predecir si un cliente suscribirá un depósito a término (`y` = `yes` / `no`).

**Plataforma:** https://ml1-competencia.pages.dev

---

## Reglas

| | |
|---|---|
| **Qué subes** | Un archivo `.pkl` con tu modelo **entrenado**, incluyendo todo el preprocesamiento. |
| **Intentos** | **5 evaluaciones exitosas** en total. Los envíos con error **no** consumen intentos. |
| **Límite de errores** | Máximo 10 envíos con error en 24 horas: valida tu modelo con `validar_modelo.py` antes de subirlo. |
| **Un envío a la vez** | Espera el resultado de un envío antes de hacer el siguiente. |
| **Cierre** | **13 de noviembre de 2026, 23:59:59 (hora de Colombia)**. El ranking sigue visible después. |
| **Ranking** | Tu mejor envío, ordenado por **precision**; empates por **F1** y luego por quién envió primero. |
| **Recall mínimo** | Solo entran al ranking los envíos con **recall ≥ 0.30**. Predecir "yes" en un único cliente muy seguro no sirve. |
| **Test oculto** | Durante la competencia ves métricas sobre el **40 %** del test oculto (ranking público). El **ranking final** usa el **60 %** restante y se revela al cierre. No sobreajustes al ranking público. |

Recibirás precision, recall, accuracy, F1 y, si tu modelo tiene `predict_proba` o `decision_function`, ROC-AUC.

## Qué debe tener tu `.pkl`

1. Un objeto con método `predict(X)`. Lo ideal es un `Pipeline` de scikit-learn.
2. `X` llega como un `DataFrame` con **las mismas columnas crudas** de los datos que te entregaron, **sin `y`**. Todo lo que hagas a los datos (imputar, codificar, escalar, crear variables) debe ir **dentro** del pipeline.
3. `predict` debe devolver `'yes'`/`'no'` o `1`/`0` (1 = suscribe).
4. Guárdalo **comprimido**: `joblib.dump(modelo, "modelo.pkl", compress=3)`.
5. Máximo 50 MB y que prediga en menos de 120 segundos.

Librerías disponibles en la plataforma: **scikit-learn, imbalanced-learn, xgboost, lightgbm**, numpy, pandas y scipy, en las versiones de [`requirements.txt`](requirements.txt). Cualquier otra librería hará fallar tu envío.

## Paso a paso

### 1. Instala las versiones exactas

**Google Colab** (en la primera celda; después reinicia el entorno: *Entorno de ejecución → Reiniciar sesión*):

```python
%pip install -q -r https://raw.githubusercontent.com/estemen27/ml1-starter-kit/main/requirements.txt
```

**En tu computador** (Python 3.12 recomendado):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # si falla xgboost-cpu, cámbialo por xgboost==3.0.2
```

### 2. Entrena

Abre [`ejemplo_entrenamiento.ipynb`](ejemplo_entrenamiento.ipynb): tiene un pipeline completo de referencia. Mejóralo.

### 3. Valida antes de subir

```bash
python validar_modelo.py modelo.pkl --datos X_train.csv --y y_train.csv
```

En Colab: descarga el script y ejecútalo **como comando** (no lo importes):

```python
!wget -q https://raw.githubusercontent.com/estemen27/ml1-starter-kit/main/validar_modelo.py
!wget -q https://raw.githubusercontent.com/estemen27/ml1-starter-kit/main/requirements.txt
!python validar_modelo.py modelo.pkl --datos X_train.csv --y y_train.csv
```

Si termina en `RESULTADO: listo para subir`, tu modelo funcionará en la plataforma.

### 4. Sube

Entra a la plataforma con el correo con el que te inscribió el profesor (te llega un enlace, sin contraseña), ve a **Mis envíos** y sube tu `.pkl`. El resultado aparece en uno o dos minutos.

## Errores frecuentes

| Mensaje | Causa y solución |
|---|---|
| *No se pudo cargar el .pkl* | Versiones distintas a las de `requirements.txt`. Instálalas, **vuelve a entrenar** y guarda de nuevo. |
| *Tu modelo usa una función o clase definida en tu notebook* | Usaste una `lambda`/función en `FunctionTransformer` o una clase propia. Usa transformadores de scikit-learn, o guarda con `cloudpickle.dump(modelo, open("modelo.pkl", "wb"))`. |
| *El modelo falló al predecir* | El pipeline espera columnas que no existen en los datos crudos (p. ej. creaste columnas o hiciste `get_dummies` fuera del pipeline). Mete ese paso dentro del pipeline. |
| *Etiquetas no reconocidas* | `predict` debe devolver `yes`/`no` o `1`/`0`. |
| *Recall menor a 0.30: no entra al ranking* | Tu modelo predice "yes" en muy pocos casos. Ajusta el umbral, usa `class_weight="balanced"` o técnicas de desbalance. |
| *El archivo pesa más de 50 MB* | Guarda con `compress=3` (un Random Forest de 185 MB queda en ~31 MB) o limita `max_depth` / `min_samples_leaf`. |
| *Superó el tiempo / la memoria* | Modelo demasiado grande (p. ej. miles de árboles muy profundos). Simplifícalo. |

¿Otro problema? Escríbele al profesor con la fecha y hora del envío.
