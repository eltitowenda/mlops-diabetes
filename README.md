[![MLOps CI](https://github.com/eltitowenda/mlops-diabetes/actions/workflows/ci.yml/badge.svg)](https://github.com/eltitowenda/mlops-diabetes/actions/workflows/ci.yml)
# Práctica 4: sistema MLOps para predecir la progresión de la diabetes

En esta práctica hemos cogido el notebook que teníamos con el dataset de diabetes de scikit-learn y lo hemos convertido en un repositorio organizado: scripts para los datos, entrenamiento con MLflow, una API con FastAPI y un workflow de GitHub Actions que lo ejecuta todo solo. No buscábamos el mejor modelo posible, sino entender cómo encaja cada pieza.

## 1. Objetivo del proyecto

Es un problema de regresión: a partir de 10 variables clínicas de un paciente (edad, índice de masa corporal, tensión...) queremos predecir cómo habrá evolucionado su diabetes al cabo de un año.

El camino que sigue el proyecto es este:

1. Extraer los datos y guardarlos sin tocar.
2. Validarlos y limpiarlos.
3. Entrenar varios modelos y guardar los resultados en MLflow.
4. Registrar el mejor y marcarlo como `champion` si cumple un mínimo de calidad.
5. Servirlo con una API.
6. Automatizar todo con GitHub Actions.

## 2. Dataset

Usamos el dataset de diabetes de scikit-learn (`sklearn.datasets.load_diabetes`). Tiene 442 pacientes y 10 variables numéricas que ya vienen estandarizadas (`age`, `sex`, `bmi`, `bp` y `s1` a `s6`). La variable que queremos predecir es `target`, un número que mide la evolución de la enfermedad.

## 3. Estructura del repositorio

```
mlops-diabetes/
├── .github/workflows/ci.yml     # workflow de GitHub Actions
├── notebooks/
│   └── Proyecto_MLOps.ipynb     # el notebook de partida
├── src/
│   ├── config.py                # rutas, constantes y umbral de calidad
│   ├── mlflow_utils.py          # configuración de MLflow
│   ├── data/
│   │   ├── extract.py           # extracción
│   │   ├── validate.py          # validación
│   │   └── clean.py             # limpieza
│   ├── training/
│   │   └── train.py             # entrenamiento y seguimiento en MLflow
│   ├── registry/
│   │   └── register.py          # registro del modelo y champion
│   ├── api/
│   │   ├── main.py              # endpoints y errores
│   │   ├── schemas.py           # validación de entrada y salida
│   │   └── service.py           # carga del modelo champion
│   └── pipeline.py              # ejecuta todos los pasos en orden
├── tests/                       # tests
├── data/                        # datos generados por el pipeline (no se suben a git)
├── reports/                     # informes generados por el pipeline
├── requirements.txt
└── README.md
```

Hemos separado cada responsabilidad en su carpeta para no meter todo en un solo archivo: datos, entrenamiento, registro de modelos, API y orquestación.

## 4. Instalación

Hace falta Python entre la 3.10 y la 3.12. Todos los comandos se lanzan desde la raíz del repositorio.

```bash
git clone <URL-de-tu-repositorio>
cd mlops-diabetes

python -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

## 5. Cómo ejecutar el pipeline

La forma más rápida es lanzar todo de una vez:

```bash
python -m src.pipeline
```

Esto ejecuta, por orden, la extracción, la validación, la limpieza, el entrenamiento y el registro del modelo. Si algún paso falla, el pipeline se para ahí.

También se puede ir paso a paso:

```bash
python -m src.data.extract         # guarda los datos originales en data/raw/
python -m src.data.validate        # comprueba el esquema de los datos
python -m src.data.clean           # guarda los datos limpios en data/processed/
python -m src.training.train       # entrena los modelos
python -m src.registry.register    # registra el mejor y decide si es champion
```

**Datos.** En la validación comprobamos que el dataset no esté vacío, que estén las 10 variables y la columna `target`, que los tipos sean numéricos y que ninguna columna tenga más de un 20 % de nulos. Si algo de esto falla se lanza un error y el pipeline no sigue. En la limpieza nos quedamos solo con las columnas esperadas, pasamos todo a numérico y quitamos las filas con nulos, duplicadas o con `target` menor o igual que 0. Con este dataset casi no hay nada que limpiar, pero el paso está para que funcione también con datos peores. Los resultados de ambos pasos quedan en `reports/`.

**Entrenamiento.** En `train.py` dividimos los datos en entrenamiento y prueba (80/20, `random_state=42`) y probamos cinco modelos: `LinearRegression`, `Ridge`, `RandomForest` y dos configuraciones de `GradientBoosting`. Cada modelo es un run de MLflow donde guardamos los parámetros, las métricas (`rmse`, `mae` y `r2`) y los artefactos (las predicciones sobre el test, las métricas en un JSON y el propio modelo).

Para ver los resultados en la interfaz de MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Se abre en <http://127.0.0.1:5000>. En *Experiments* están los runs y en *Models* el modelo registrado con sus versiones y el alias `champion`.

## 6. Cómo se elige el champion

1. De los cinco modelos nos quedamos con el que tiene menor RMSE (raíz del error cuadrático medio) sobre el conjunto de prueba.
2. `register.py` lo registra en el Model Registry de MLflow con el nombre `diabetes-progression-regressor`. Cada registro crea una versión nueva.
3. Después aplicamos un *quality gate*: para ser champion, el RMSE tiene que ser menor que 60. Elegimos 60 porque los modelos que probamos dan entre 52 y 55, y la desviación típica de `target` es de unos 77. Así nos aseguramos de que el modelo mejora bastante a predecir siempre la media, sin poner un listón tan justo que falle por cualquier pequeña variación.
4. Si ya había un champion, comparamos: si el nuevo es igual o mejor, se queda con la etiqueta `champion`; si es peor, seguimos con el anterior.
5. Si no pasa el quality gate, el modelo se registra igualmente pero sin etiqueta, y el script termina con error. Así el workflow se pone en rojo y nos damos cuenta.

## 7. La API

Primero hay que haber ejecutado el pipeline, para que exista un champion. Después:

```bash
uvicorn src.api.main:app --reload
```

La API queda en <http://127.0.0.1:8000>, y en <http://127.0.0.1:8000/docs> hay una documentación interactiva desde la que se pueden probar los endpoints sin usar la terminal.

Al arrancar intenta cargar el modelo con el alias `champion`. Si no lo encuentra (por ejemplo, porque todavía no hemos entrenado nada), arranca igualmente, pero `/predict` y `/model-info` darán un 503. Entrenando y reiniciando la API se arregla.

### Probar los endpoints

```bash
# comprobar que el servicio está vivo
curl http://127.0.0.1:8000/health
# {"status":"ok"}

# ver qué modelo se está sirviendo (nombre, versión, alias, métricas...)
curl http://127.0.0.1:8000/model-info

# pedir una predicción
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.0381,"sex":0.0507,"bmi":0.0617,"bp":0.0219,"s1":-0.0442,"s2":-0.0348,"s3":-0.0434,"s4":-0.0026,"s5":0.0199,"s6":-0.0176}'
# {"prediction":...,"model_name":"diabetes-progression-regressor","model_version":"1"}
```

Las variables van en la escala estandarizada del dataset (valores entre −0.2 y 0.2 más o menos), no en unidades reales. En Windows PowerShell es más cómodo usar la página `/docs`.

### Errores que gestiona la API

- **200**: la petición es correcta y devuelve la predicción y la versión del modelo.
- **422**: faltan campos, hay un tipo incorrecto (por ejemplo `"age": "abc"`) o llega un campo que no existe. Lo valida Pydantic.
- **503**: el modelo champion no está cargado. `/health` sigue respondiendo 200 porque el servicio en sí está funcionando.
- **500**: algo falla al predecir. Se guarda el error en el log y se devuelve un mensaje genérico.

Para ver un 422 basta con mandar una petición incompleta:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"age":0.05}'
```

## 8. CI/CD con GitHub Actions

El workflow está en `.github/workflows/ci.yml` y se ejecuta solo cada vez que hacemos `push` a `main` o `master`. Los pasos son:

1. Instalar Python 3.11 y las dependencias.
2. Pasar los tests (`python -m pytest`).
3. Ejecutar el pipeline en este orden: extracción, validación, limpieza, entrenamiento y registro.
4. Probar la API con el champion real.
5. Guardar los resultados de MLflow y los informes como artefactos, para poder descargarlos desde la pestaña *Actions*.

Si algo falla (datos inválidos, un modelo que no pasa el quality gate, un test roto...), el workflow se pone en rojo.

Un detalle: la base de datos de MLflow (`mlflow.db`) se crea desde cero en cada ejecución, así que en GitHub Actions el registro desaparece cuando termina el job. Para algo en producción habría que usar un servidor de MLflow permanente, pero para esta práctica nos basta con comprobar que el pipeline funciona de principio a fin.
