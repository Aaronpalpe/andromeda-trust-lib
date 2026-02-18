import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import mean_squared_error, roc_curve
from sklearn.model_selection import train_test_split

from .utils import Result, calculate_score

from holisticai.security.metrics import k_anonymity, l_diversity, attribute_attack_score, data_minimization_score, privacy_risk_score, shapr_score

    
def load_privacy_config(factsheet):
    try:
        privacy_section = factsheet.get("privacy", {})
        general_section = factsheet.get("general", {})

        epsilon = privacy_section.get("epsilon", {}).get("value")
        sensitive_attribute = privacy_section.get("sensitive_attribute", {}).get("value")
        target_column = general_section.get("target_column", {}).get("value")
        quasi_identifiers = privacy_section.get("quasi_identifiers", {}).get("value")

        return epsilon, quasi_identifiers, sensitive_attribute, target_column

    except Exception as e:
        raise ValueError("Privacy configuration incomplete.")

def analyse(context, config):
    epsilon, quasi_identifiers, sensitive_attribute, target_column = load_privacy_config(context.factsheet)
    
    def get_thresh(key):
        val = config.get(key, {}).get("thresholds", {}).get("value")
        return val if val is not None else [0.1, 0.2, 0.3, 0.4]

    output = {
        "epsilon_dp": epsilon_dp_score(context, get_thresh("score_epsilon_dp")),
        "epsilon_star": epsilon_star_score(context, get_thresh("score_epsilon_star")),
        "shapr": shapr_mod_score(context, get_thresh("score_shapr")),
        "attribute_inference": attribute_inference_score(context, get_thresh("score_attribute_inference")),
        "accuracy_ratio": accuracy_ratio_score(context, get_thresh("score_accuracy_ratio")),
        "privacy_risk": privacy_score(context, get_thresh("score_privacy_risk")),
        "k_anonymity": k_anonymity_score(context, get_thresh("score_k_anonymity")),
        "l_diversity": l_diversity_score(context, get_thresh("score_l_diversity")),
        "t_closeness": t_closeness_score(context, get_thresh("score_t_closeness"))
    }

    scores = {k: v.score for k, v in output.items()}
    properties = {k: v.properties for k, v in output.items()}

    return Result(score=scores, properties=properties)


def epsilon_dp_score(context, thresholds):
    try:
        epsilon, _, _, _ = load_privacy_config(context.factsheet)

        if epsilon is None:
            return Result(np.nan, {"Info": "No epsilon provided in factsheet."})

        score = calculate_score(epsilon, thresholds)

        props = {
            "Metric Description": "Differential Privacy parameter epsilon.",
            "Depends on": "Training Mechanism",
            "Epsilon": epsilon,
            "Interpretation": (
                "Lower epsilon implies stronger privacy guarantees."
            )
        }

        return Result(score, props)
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})

def calculate_losses(model, X, y):
    """
    Compute Log Loss (Cross-Entropy) for each instance.
    """
    # Obtener probabilidades (n_samples, n_classes)
    probs = model.predict_proba(X)
    
    # Mapear las etiquetas reales a los índices de las columnas de probs
    if hasattr(model, 'classes_'):
        class_map = {c: i for i, c in enumerate(model.classes_)}
        y_indices = np.array([class_map[label] for label in y])
    else:
        # Fallback asumiendo índices directos
        y_indices = y.astype(int)
        
    # Extraer la probabilidad asignada a la clase correcta
    true_class_probs = probs[np.arange(len(y)), y_indices]
    
    # Clipping para evitar log(0)
    true_class_probs = np.clip(true_class_probs, 1e-15, 1.0 - 1e-15)
    
    # Loss = -log(p)
    return -np.log(true_class_probs)

def epsilon_star_score(context, thresholds):
    """
    Compute empirical epsilon* based on Loss Distribution as Definition 2 of the paper.
    """
    try:
        _, _, _, target = load_privacy_config(context.factsheet)


        # 1. Calcular Losses. Los miembros (train) suelen tener menor loss que los no-miembros (test).
        loss_train = calculate_losses(context.model, context.X_train, context.y_train)
        loss_test = calculate_losses(context.model, context.X_test, context.y_test)

        # 2. Configurar el problema como clasificación binaria para obtener TPR y FPR
        scores = np.concatenate([-loss_train, -loss_test])
        
        # Etiquetas: 1 = Miembro (Train), 0 = No Miembro (Test)
        y_true = np.concatenate([np.ones(len(loss_train)), np.zeros(len(loss_test))])

        # 3. Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_true, scores)
        
        # Limpieza numérica para evitar divisiones por cero o log(0)
        fpr = np.clip(fpr, 1e-10, 1.0 - 1e-10)
        tpr = np.clip(tpr, 1e-10, 1.0 - 1e-10)

        # Calculamos el fnr (False Negative Rate) a partir de tpr
        fnr = 1 - tpr

        # 4. Definir Delta (delta). El paper recomienda delta << 1/n.
        n_train = len(loss_train)
        delta = 1.0 / n_train if n_train > 0 else 1e-5

        # 5. Calcular los 4 términos de la Definición 2. Paper usa notación: t = FPR, eta (η) = FNR = 1 - TPR
        m1 = (1 - delta - fnr) / fpr
        m2 = (1 - delta - fpr) / fnr
        m3 = (fnr - delta) / (1 - fpr)
        m4 = (fpr - delta) / (1 - fnr)
        
        # Epsilon* es el máximo valor encontrado en cualquier umbral
        epsilon_star_val = np.log(np.nanmax(np.maximum.reduce([m1, m2, m3, m4, np.ones_like(m1)])))
        
        score = calculate_score(epsilon_star_val, thresholds)

        props = {
            "Metric Description": "Empirical epsilon* derived from Loss Distribution Hypothesis Test.",
            "Delta Used": f"{delta:.2e}",
            "Max Term Value": f"{np.exp(epsilon_star_val):.4f}", # e^epsilon
            "Epsilon*": f"{epsilon_star_val:.4f}",
            "Interpretation": "Lower is better (more private). Represents privacy loss."
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})



def shapr_mod_score(context, thresholds):
    '''
    Compute an approximate SHAPr score for membership inference risk.
    '''
    try:
        _, _, _, target = load_privacy_config(context.factsheet)

        sample_size = min(5000, len(context.X_train), len(context.X_test))
        # Por ejemplo, tomar un subconjunto aleatorio
        train_indices = np.random.choice(len(context.X_train), size=sample_size, replace=False)
        test_indices = np.random.choice(len(context.X_test), size=sample_size, replace=False)
        
        # 3. Apply the specific indices to their respective datasets
        X_train_sample = context.X_train.iloc[train_indices]
        y_train_sample = context.y_train[train_indices]

        X_test_sample = context.X_test.iloc[test_indices]
        y_test_sample = context.y_test[test_indices]

        y_pred_train = context.model.predict(X_train_sample)
        y_pred_test = context.model.predict(X_test_sample)

        # It fits a k-nearest neighbors (KNN) classifier using the training predictions 
        # as input and the true labels as output. Then, for each batch of test samples, 
        # it obtains the nearest neighbor indices and calculates binary indicators that 
        # represent whether the neighbor labels match the test labels. On these indicators, 
        # a normalized cumulative difference (d_phi_y) and its cumulative sum (phi_y) are computed, 
        # which are reordered according to the original neighbors; finally, the results from all batches 
        # are aggregated and normalized to obtain an aggregated value per sample that indicates 
        # the privacy leakage risk, where higher values ​​indicate higher risk.
        phi = shapr_score(y_train_sample, y_test_sample, y_pred_train, y_pred_test)

        phi = float(np.mean(phi))  

        score = calculate_score(phi, thresholds)

        props = {
            "Metric Description": "Approximate SHAPr membership risk.",
            "Average Marginal Contribution": f"{phi:.6f}"
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})

def attribute_inference_score(context, thresholds):
    try:
        _, _, sensitive_attr, target = load_privacy_config(context.factsheet)

        if sensitive_attr is None:
            return Result(np.nan, {"Info": "No sensitive attribute defined."})

        # Tomamos el primer elemento de la lista
        sensitive_attr = sensitive_attr[0]

        # Asegurarse de que la columna exista
        if sensitive_attr not in context.train_data.columns:
            return Result(
                np.nan,
                {"Error": f"Sensitive attribute '{sensitive_attr}' not found in DataFrame."}
            )
        # The library function 'to_numerical_or_categorical' requires pandas objects
        # to use .astype("category")
        y_train_series = pd.Series(context.y_train)
        y_test_series = pd.Series(context.y_test)

        # If an attacker estimator is not provided, it automatically selects a linear regressor
        # for continuous attributes or a logistic classifier for categorical attributes,
        # also assigning the appropriate metric function (mean squared error for continuous,
        # precision or F1 for categorical).
        # It then creates a BlackBoxAttack object that removes the target attribute from
        # the training set, adds the label as a new feature, and trains the attacker model.
        # Subsequently, it uses this model to predict the removed attribute on the test set
        # and compares the predictions with the actual values using the selected metric
        # function.
        # Finally, it returns a numerical value indicating how predictable the attribute
        # was based on the other features and labels, with higher values indicating a greater
        # risk of information leakage for that attribute.
        res = attribute_attack_score(
            context.X_train,
            context.X_test,
            y_train_series,
            y_test_series,
            sensitive_attr
        )

        # Detectamos si el atributo es continuo
        is_continuous = context.X_train[sensitive_attr].dtype.kind in ["i", "u", "f"]

        # Si es continuo, invertimos el MSE para que score alto = mayor riesgo
        if is_continuous:
            res = 1 / (1 + res)  # ahora valores más altos indican mayor riesgo

        score = calculate_score(res, thresholds)

        props = {
            "Metric": "Attribute Inference",
            "Sensitive Attribute": sensitive_attr,
            "Accuracy score": f"{res:.6f}"
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})



def accuracy_ratio_score(context, thresholds):
    try:
        _, _, _, target = load_privacy_config(context.factsheet)

        X_noisy = context.X_test + np.random.normal(0, 0.01, context.X_test.shape)
        y_pred_noisy = context.model.predict(X_noisy)

        # Preparar diccionario para data minimization
        y_pred_dm = [
            {
                "selector_type": "Noisy",
                "modifier_type": "GaussianNoise",
                "n_feats": context.X_test.shape[1],
                "feats": [f"feat_{i}" for i in range(context.X_test.shape[1])],
                "predictions": y_pred_noisy,  # ya es ndarray
            }
        ]

        # Evaluate how data or feature reduction affects a model's predictive ability. 
        # For each version of the minimized model (`y_pred_dm`), the corresponding prediction 
        # is calculated and compared to the prediction of the full model (`y_pred`) using a relative metric. 
        # The lowest score among all techniques is selected as the relative performance indicator. 
        # Values ​​close to 1 indicate that data reduction maintains similar performance to the full model.
        ratio = data_minimization_score(context.y_test, context.y_pred_test, y_pred_dm)

        score = calculate_score(ratio, thresholds)

        props = {
            "Metric Description": "Utility retention after privacy mechanism.",
            # "Original Accuracy": f"{ratio:.2%}",
            # "After Minimization Accuracy": f"{score:.2%}",
            "Accuracy Ratio": f"{ratio:.4f}"
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})
    

def privacy_score(context, thresholds):
    """
    Compute the privacy risk score based on membership inference attacks.
    This metric estimates the likelihood that a training sample belongs to the 
    model's training set by analyzing the model's behavior. Higher scores indicate
    higher risk of privacy leakage.
    """
    try:
        _, _, _, target_column = load_privacy_config(context.factsheet)

        # Preparar tuplas para la función privacy_risk_score
        shadow_train = (context.y_prob_train, context.y_train)
        shadow_test = (context.y_prob_test, context.y_test)
        target_train = (context.y_prob_train, context.y_train)

        # Calcular los scores de riesgo de privacidad
        scores_array = privacy_risk_score(shadow_train, shadow_test, target_train)

        # Promediar scores y normalizar según thresholds
        mean_score = float(np.mean(scores_array))
        # La métrica calcula qué tan probable es que una muestra haya estado en el conjunto de entrenamiento de un modelo. 
        # Para ello, transforma las predicciones en entropía modificada que refleja incertidumbre, construye histogramas 
        # normalizados de entrenamiento y prueba por clase usando un shadow model, y para cada muestra calcula un score 
        # como la proporción de su probabilidad en entrenamiento sobre la suma de entrenamiento y prueba. Un score alto 
        # indica que la muestra es más predecible y, por tanto, hay mayor riesgo de un ataque de inferencia de membresía.
        score = calculate_score(mean_score, thresholds)

        props = {
            "Metric Description": "Privacy risk based on membership inference.",
            "Mean Privacy Risk": f"{mean_score:.6f}",
            "Thresholds Used": thresholds
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})

def k_anonymity_score(context, thresholds):
    """
    Compute k-anonymity for a dataset.
    """
    try:
        _, quasi_identifiers, _, _ = load_privacy_config(context.factsheet)

        # Unimos los datasets para calcular k-anonymity globalmente
        combined_df = pd.concat([context.train_data, context.test_data], ignore_index=True)
        
        counts = combined_df[quasi_identifiers].value_counts() # Cuenta cuántas veces aparece cada combinación de quasi-identificadores
        k_value = counts.min() if not counts.empty else 0

        score = calculate_score(k_value, thresholds)

        props = {
            "Metric Description": "k-Anonymity measures the minimum number of identical quasi-identifiers.",
            "Quasi Identifiers": quasi_identifiers,
            "Minimum k": k_value
        }
        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def l_diversity_score(context, thresholds):
    """
    Compute l-diversity for sensitive attributes in a dataset.
    """
    try:
        _, quasi_identifiers, sensitive_attributes, _ = load_privacy_config(context.factsheet)
        df = pd.concat([context.train_data, context.test_data], ignore_index=True)

        df_grouped = df.groupby(quasi_identifiers, as_index=False)
        l_div = {
            s: sorted([len(row["unique"]) for _, row in df_grouped[s].agg(["unique"]).dropna().iterrows()])
            for s in sensitive_attributes
        }

        # Tomamos el valor mínimo de l-diversity como score global
        min_l = min([min(v) if v else 0 for v in l_div.values()])
        score = calculate_score(min_l, thresholds) if thresholds is not None else min_l

        props = {
            "Metric Description": "l-Diversity measures diversity of sensitive attributes for each quasi-identifier group.",
            "Quasi Identifiers": quasi_identifiers,
            "Sensitive Attributes": sensitive_attributes,
            #"l-Diversity per Attribute": l_div,
            "Minimum l": min_l
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def t_closeness_score(context, thresholds):
    """
    Compute t-closeness for sensitive attributes using Earth Mover's Distance (EMD) between local and global distributions.
    """
    try:
        _, quasi_identifiers, sensitive_attributes, _ = load_privacy_config(context.factsheet)
        df = pd.concat([context.train_data, context.test_data], ignore_index=True)
        results = {}

        for s in sensitive_attributes:
            global_dist = df[s].value_counts(normalize=True)
            distances = []

            grouped = df.groupby(quasi_identifiers)
            for _, group in grouped:
                local_dist = group[s].value_counts(normalize=True)
                aligned = global_dist.index.union(local_dist.index)
                g = global_dist.reindex(aligned, fill_value=0)
                l = local_dist.reindex(aligned, fill_value=0)

                # tvd = 0.5 * np.sum(np.abs(g - l)) # Total Variation Distance.
                # distances.append(tvd)
                # Orden importante si es ordinal o numérico
                g_cdf = np.cumsum(g.values)
                l_cdf = np.cumsum(l.values)

                emd = np.sum(np.abs(g_cdf - l_cdf))
                distances.append(emd)


            results[s] = sorted(distances)

        # Tomamos el valor máximo como score global
        max_t = max([max(v) if v else 0 for v in results.values()])
        score = calculate_score(max_t, thresholds) if thresholds is not None else max_t

        props = {
            "Metric Description": "t-Closeness measures deviation of sensitive attribute distributions from the global distribution.",
            "Quasi Identifiers": quasi_identifiers,
            "Sensitive Attributes": sensitive_attributes,
            #"t-Closeness per Attribute": results,
            "Maximum t": max_t
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})
