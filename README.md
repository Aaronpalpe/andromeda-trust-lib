# Andrómeda Lib

Librería para evaluar la confianza de modelos de machine learning a partir de varios pilares. El proyecto agrupa métricas y utilidades para analizar:

- Fairness
- Accountability
- Privacy
- Sustainability
- Explainability
- Robustness

El punto de entrada principal es `TrustEvaluator`, que calcula una puntuación global de confianza, el desglose por pilar, el detalle de métricas y una explicación del resultado. El proyecto también incluye un wrapper para usar modelos de distintos frameworks como estimadores tipo scikit-learn.

## Estructura principal

- `src/trust_library/`: código de la librería
- `main.ipynb`: notebook principal de trabajo y reproducción
- `models_and_data/`: datasets, particiones y modelos guardados usados en los experimentos
- `UniversarSklearnWrapper.py`: adaptador para modelos de scikit-learn, PyTorch y TensorFlow
- `requirements.txt`: dependencias del entorno de ejecución

## Requisitos

Se recomienda usar Python 3.12.1 y crear un entorno aislado con Conda. 

**Importante**

Aunque la librería sigue en desarrollo y está pensada para ser utilizada de forma independiente del sistema operativo, actualmente algunas dependencias presentan incompatibilidades de versiones de Python en entornos Linux.
Para poder ejecutar los notebooks y reproducir fielmente los experimentos, se recomienda utilizar Windows.

## Instalación y reproducción de experimentos

Desde la raíz del repositorio:

```bash
conda create --name environment python=3.12.1 --no-default-packages
conda activate environment
pip install -r requirements.txt
```

Si quieres trabajar con la librería localmente desde notebooks o scripts, abre el proyecto desde la raíz del repositorio para que `src/` y los ficheros del proyecto queden accesibles.

## Uso básico

```python
from trust_library import TrustEvaluator
from UniversarSklearnWrapper import UniversalSklearnWrapper

# model: modelo ya entrenado
# train_data / test_data: pandas.DataFrame o estructuras compatibles
# factsheet: instancia Factsheet con la información del modelo

evaluator = TrustEvaluator(
    model=model,
    train_data=train_data,
    test_data=test_data,
    factsheet=factsheet,
)

result = evaluator.evaluate()
evaluator.plot_results()
```

## Notas

- Los experimentos y ejemplos dependen de los datos y modelos almacenados en `models_and_data/`.
- La librería está pensada para evaluar modelos sobre datos tabulares o clasificadores ya entrenados.