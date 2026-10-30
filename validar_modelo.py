#!/usr/bin/env python3
"""Valida tu modelo ANTES de subirlo a la competencia ML1.

Uso:
    python validar_modelo.py modelo.pkl --datos X_train.csv --y y_train.csv

Revisa lo mismo que la plataforma: versiones de las librerías, que el .pkl cargue,
que tenga predict(), que reciba las columnas crudas (sin 'y') y que devuelva
etiquetas válidas. Con --y muestra métricas sobre ESOS datos: NO son las
métricas de la competencia (esas usan datos ocultos).

Ejecútalo como script (en Colab: !python validar_modelo.py ...), no lo importes:
así se comporta igual que la plataforma, que carga tu modelo en un proceso nuevo.
"""

import argparse
import importlib.metadata as metadata
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_BYTES = 50 * 1024 * 1024
MIN_RECALL = 0.30
TIME_LIMIT_S = 120
CRITICAL = {"scikit-learn", "numpy", "joblib"}
DISTRIBUTIONS = {"xgboost-cpu": ("xgboost-cpu", "xgboost")}  # la plataforma usa xgboost-cpu; en tu PC sirve xgboost
POSITIVE = {"yes", "si", "sí", "true", "1", "1.0"}
NEGATIVE = {"no", "false", "0", "0.0"}


class Report:
    def __init__(self):
        self.errors = 0
        self.warnings = 0

    def ok(self, msg):
        print(f"  [OK]     {msg}")

    def info(self, msg):
        print(f"  [INFO]   {msg}")

    def warn(self, msg):
        self.warnings += 1
        print(f"  [AVISO]  {msg}")

    def error(self, msg):
        self.errors += 1
        print(f"  [ERROR]  {msg}")


def to_binary(label):
    text = str(label).strip().lower()
    if text in POSITIVE:
        return 1
    if text in NEGATIVE:
        return 0
    return None


def required_versions():
    pins = {}
    for line in (HERE / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, version = line.split("==")
            pins[name.strip()] = version.strip()
    return pins


def installed_version(name):
    for dist in DISTRIBUTIONS.get(name, (name,)):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return None


def check_versions(r):
    print("\n1. Versiones de librerías (deben coincidir con las de la plataforma)")
    for name, want in required_versions().items():
        have = installed_version(name)
        pip_name = "xgboost" if name == "xgboost-cpu" else name
        if have is None:
            if name in CRITICAL:
                r.error(f"{name} no está instalado. Instala con: pip install {pip_name}=={want}")
            else:
                r.info(f"{name} no instalado (no lo necesitas si tu modelo no lo usa)")
        elif have == want:
            r.ok(f"{name} {have}")
        elif name in CRITICAL:
            r.error(f"{name} {have} instalado; la plataforma usa {want} y tu .pkl puede no cargar. "
                    f"Instala con: pip install {pip_name}=={want} y vuelve a entrenar y guardar.")
        else:
            r.warn(f"{name} {have} instalado; la plataforma usa {want}. Si tu modelo lo usa, iguala la versión.")


def load_model(r, path):
    print("\n2. Archivo y carga del modelo")
    size = path.stat().st_size
    if size > MAX_BYTES:
        r.error(f"El archivo pesa {size / 1048576:.1f} MB; el máximo es 50 MB. Guárdalo comprimido: "
                "joblib.dump(modelo, 'modelo.pkl', compress=3) (un Random Forest de 185 MB queda en ~31 MB), "
                "o limita el modelo (max_depth, min_samples_leaf, menos árboles).")
    else:
        shown = f"{size / 1048576:.1f} MB" if size >= 1048576 else f"{max(1, round(size / 1024))} KB"
        r.ok(f"Tamaño {shown} (máximo 50 MB)")

    import joblib

    try:
        model = joblib.load(path)
    except ModuleNotFoundError as exc:
        r.error(f"Tu modelo necesita la librería '{exc.name}', que la plataforma no tiene. "
                "Usa scikit-learn, imbalanced-learn, xgboost o lightgbm.")
        return None
    except Exception as exc:
        text = f"{type(exc).__name__}: {exc}"
        if "__main__" in text:
            r.error("Tu modelo usa una función o clase definida en tu notebook (por ejemplo una lambda en "
                    "FunctionTransformer o una clase propia). La plataforma no tiene ese código. "
                    "Opciones: usa solo transformadores de scikit-learn, o guarda con cloudpickle: "
                    "import cloudpickle; cloudpickle.dump(modelo, open('modelo.pkl', 'wb'))")
        else:
            r.error(f"No se pudo cargar el .pkl: {text}")
        return None
    r.ok(f"Cargó correctamente: {type(model).__name__}")

    if not hasattr(model, "predict"):
        r.error("El objeto no tiene método predict(). Guarda el modelo o Pipeline ya entrenado.")
        return None
    r.ok("Tiene predict()")
    return model


def positive_scores(model, X, np):
    classes = getattr(model, "classes_", None)
    if classes is None or len(classes) != 2:
        return None, None
    pos = [i for i, c in enumerate(classes) if to_binary(c) == 1]
    if len(pos) != 1:
        return None, None
    for method in ("predict_proba", "decision_function"):
        if hasattr(model, method):
            try:
                values = np.asarray(getattr(model, method)(X), dtype=float)
                if method == "predict_proba":
                    return values[:, pos[0]], method
                values = values.ravel()
                return (values if pos[0] == 1 else -values), method
            except Exception:
                continue
    return None, None


def check_predictions(r, model, datos, etiquetas=None):
    from io import StringIO

    import numpy as np
    import pandas as pd

    print("\n3. Predicción con columnas crudas (como en la plataforma)")
    X = pd.read_csv(datos, sep=None, engine="python")
    y = X.pop("y") if "y" in X.columns else None
    if etiquetas is not None:
        y = pd.read_csv(etiquetas)["y"]
        if len(y) != len(X):
            r.error(f"{etiquetas} tiene {len(y)} filas y {datos} tiene {len(X)}.")
            return
    # La plataforma entrega un CSV estándar: se replica ese viaje para tener los mismos tipos de dato.
    X = pd.read_csv(StringIO(X.to_csv(index=False)))
    r.info(f"{len(X)} filas, {X.shape[1]} columnas: {', '.join(map(str, X.columns[:8]))}{'…' if X.shape[1] > 8 else ''}")

    start = time.perf_counter()
    try:
        raw = np.asarray(model.predict(X)).ravel()
    except Exception as exc:
        r.error(f"predict() falló: {type(exc).__name__}: {exc}. Revisa que el Pipeline incluya TODO el "
                "preprocesamiento y que no dependa de columnas creadas fuera de él.")
        return
    elapsed = time.perf_counter() - start
    if elapsed > TIME_LIMIT_S / 4:
        r.warn(f"predict() tardó {elapsed:.1f} s; la plataforma corta a los {TIME_LIMIT_S} s.")
    else:
        r.ok(f"predict() tardó {elapsed:.2f} s")

    if len(raw) != len(X):
        r.error(f"predict() devolvió {len(raw)} filas y se esperaban {len(X)}.")
        return
    pred = [to_binary(v) for v in raw]
    unknown = sorted({str(v) for v, p in zip(raw, pred) if p is None})
    if unknown:
        r.error(f"Etiquetas no reconocidas: {unknown[:5]}. Usa 'yes'/'no' o 1/0.")
        return
    pred = np.array(pred)
    r.ok(f"Etiquetas válidas; predice 'yes' en el {pred.mean():.1%} de las filas")
    if pred.min() == pred.max():
        r.warn("Predice la misma clase para todas las filas.")

    score, source = positive_scores(model, X, np)
    if score is not None:
        r.ok(f"ROC-AUC disponible (usa {source})")
    else:
        r.info("Sin predict_proba ni decision_function: ROC-AUC no se calculará")

    if y is None:
        r.info("--datos no tiene columna 'y': no se calculan métricas")
        return
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y_true = [to_binary(v) for v in y]
    if any(v is None for v in y_true):
        r.warn("La columna 'y' tiene valores no reconocidos; se omiten las métricas.")
        return
    y_true = np.array(y_true, dtype=int)
    recall = recall_score(y_true, pred, zero_division=0)
    print("\n4. Métricas sobre TUS datos (orientativas: la competencia usa datos ocultos)")
    print(f"     precision {precision_score(y_true, pred, zero_division=0):.4f}   recall {recall:.4f}   "
          f"f1 {f1_score(y_true, pred, zero_division=0):.4f}   accuracy {accuracy_score(y_true, pred):.4f}"
          + (f"   roc_auc {roc_auc_score(y_true, score):.4f}" if score is not None else ""))
    if recall < MIN_RECALL:
        r.warn(f"Recall {recall:.2f} < {MIN_RECALL}: con este comportamiento tu envío NO entraría al ranking.")
    print("     Si son las mismas filas con las que entrenaste, estas métricas son optimistas.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("modelo", type=Path, help="tu archivo .pkl")
    parser.add_argument("--datos", type=Path, required=True, help="CSV con las columnas crudas, p. ej. X_train.csv")
    parser.add_argument("--y", type=Path, help="CSV con la columna 'y', p. ej. y_train.csv (para ver métricas)")
    args = parser.parse_args()
    for path in (args.modelo, args.datos, args.y):
        if path is not None and not path.is_file():
            sys.exit(f"No existe el archivo: {path}")

    r = Report()
    check_versions(r)
    model = load_model(r, args.modelo)
    if model is not None:
        check_predictions(r, model, args.datos, args.y)

    print()
    if r.errors:
        print(f"RESULTADO: {r.errors} error(es). Corrígelos antes de subir: en la plataforma este modelo fallaría.")
        return 1
    print("RESULTADO: listo para subir." + (f" Revisa los {r.warnings} aviso(s)." if r.warnings else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
