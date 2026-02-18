import numpy as np
import collections
import re
import tensorflow as tf
from math import isclose

from sklearn import metrics

from trust_library.factsheet import load_factsheet_default
from .utils import Result  

info = collections.namedtuple('info', 'description value')

# === MAIN ANALYSE ===

def analyse(context, methodology_config):

    output = {
        "train_test_split": train_test_split_score(
            context, methodology_config["score_train_test_split"]["mappings"]["value"]
        ),
        "missing_data": missing_data_score(
            context, methodology_config["score_missing_data"]["mappings"]["value"]
        ),
        "normalization": normalization_score(
            context, methodology_config["score_normalization"]["mappings"]["value"]
        ),
        "regularization": regularization_score(
            context, None
        ),
        "factsheet_completeness": factsheet_completeness_score(
            context, None
        )
    }

    scores = {k: v.score for k, v in output.items()}
    properties = {k: v.properties for k, v in output.items()}

    return Result(score=scores, properties=properties)

# === NORMALIZATION ===

def normalization_score(context, mappings):
    try:
        X_train = context.X_train
        X_test = context.X_test

        train_mean = np.mean(X_train.values)
        train_std = np.std(X_train.values)
        test_mean = np.mean(X_test.values)
        test_std = np.std(X_test.values)

        properties = {
            "Depends on": "Training and Testing Data",
            "Training mean": f"{train_mean:.2f}",
            "Training std": f"{train_std:.2f}",
            "Test mean": f"{test_mean:.2f}",
            "Test std": f"{test_std:.2f}"
        }
                        
        # Diferente al original
        if (isclose(train_mean, 0, abs_tol=1e-3) and isclose(train_std, 1, abs_tol=1e-3) and 
            isclose(test_mean, 0, abs_tol=1e-3) and isclose(test_std, 1, abs_tol=1e-3)): 
            score = mappings["training_and_test_standardize"]
            properties["Conclusion"] = "Training and Testing data are standardized"
        
        elif (isclose(train_mean, 0, abs_tol=1e-3) and isclose(train_std, 1, abs_tol=1e-3)):
            score = mappings["training_standardized"] 
            properties["Conclusion"] = "Training data are standardized"
        
        elif (X_train.min().min() >= 0 and X_train.max().max() <= 1 and
            X_test.min().min() >= 0 and X_test.max().max() <= 1):
            score = mappings["training_and_test_normal"]
            properties["Conclusion"] = "Training and Testing data are normalized"

        elif (X_train.min().min() >= 0 and X_train.max().max() <= 1): 
            score = mappings["training_normal"] 
            properties["Conclusion"] = "Training data are normalized"

        else:
            score = mappings["None"]
            properties["Conclusion"] = "No normalization detected"

        return Result(score, properties)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})

# === MISSING DATA ===

def missing_data_score(context, mappings):
    try:
        missing_values = (
            context.train_data.isna().sum().sum() +
            context.test_data.isna().sum().sum()
        )

        score = (
            mappings["null_values_exist"]
            if missing_values > 0
            else mappings["no_null_values"]
        )

        props = {
            "Depends on": "Training and Test Data",
            "Null values count": missing_values,
            "Conclusion": (
                "Missing values present in train and/or test data" if missing_values > 0
                else "No missing values"
            )
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})

# === TRAIN / TEST SPLIT ===

def train_test_split_score(context, mappings):
    try:
        n_train = len(context.train_data)
        n_test = len(context.test_data)
        total = n_train + n_test

        train_ratio = round(n_train / total * 100)
        test_ratio = round(n_test / total * 100)

        score = np.nan
        for k, v in mappings.items():
            bounds = re.findall(r'\d+-\d+', k)
            for b in bounds:
                low, high = map(int, b.split("-"))
                if low <= train_ratio < high:
                    score = v

        props = {
            "Depends on": "Training and Testing Data",
            "Train/Test split": f"{train_ratio}/{test_ratio}"
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})

# === REGULARIZATION ===

def regularization_score(context, mappings):
    try:
        reg = context.factsheet.get("methodology", {}).get("regularization", None)

        mapping = {
            "elasticnet_regression": 5,
            "lasso_regression": 4,
            "ridge_regression": 4,
            "other": 3,
            None: 0
        }

        score = mapping.get(reg, 1)  # 1 = técnica desconocida pero documentada

        props = {
            "Depends on": "Factsheet",
            "Regularization technique": reg if reg is not None else "Not specified"
        }

        return Result(score, props)

    except Exception as e:
        return Result(0, {"Error": str(e)})


# === FACTSHEET COMPLETENESS ===

def is_present(value):
    return value not in (None, "", [], {})

def factsheet_completeness_score(context, mappings):
    """
    Analiza la completitud de la factsheet verificando si los campos tienen
    un valor asignado en su clave 'value'.    
    """
    try:
        total = 0
        present = 0
        missing_fields_log = []

        for section, fields in context.factsheet.items():
            for field_key, field_data in fields.items():
                total += 1
                value = field_data.get("value")
                
                is_filled = False
                
                if value is not None:
                    if isinstance(value, (str, list, dict)):
                        # Si es texto, lista o dict, verificamos que no esté vacío
                        if len(value) > 0:
                            is_filled = True
                    else:
                        # Números (int, float) o booleanos
                        is_filled = True

                if is_filled:
                    present += 1
                else:
                    # Guardamos el nombre del campo faltante para mostrarlo
                    missing_fields_log.append(f"{section}.{field_key}")

        completeness_ratio = present / total if total > 0 else 0
        score = round(completeness_ratio * 5)

        props = {
            "Depends on": "Factsheet",
            "Fields present": f"{present}/{total}",
            "Completeness": f"{completeness_ratio:.2%}",
            "Missing fields": missing_fields_log if missing_fields_log else "None"
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})
