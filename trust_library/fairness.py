import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.stats import chisquare
from .utils import Result, calculate_score

# AIF360 Imports
try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
except ImportError:
    raise ImportError("La librería 'aif360' es necesaria. Instálala con 'pip install aif360'")

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from holisticai.bias.metrics import classification_bias_metrics

import warnings
warnings.filterwarnings("ignore")
# === CONFIG & AIF360 HELPERS ====

def load_fairness_config(factsheet):
    '''
    Carga la configuración de equidad (fairness) desde la factsheet
    '''
    try:
        fairness_section = factsheet.get("fairness", {})
        general_section = factsheet.get("general", {})

        get_val = lambda section_dict, field, default: section_dict.get(field, {}).get("value") or default

        protected_feature = get_val(fairness_section, "protected_feature", "")

        # Aseguramos que protected_values sea una lista, a veces viene como int único
        raw_protected_values = fairness_section.get("protected_values", {}).get("value")
        
        if raw_protected_values is None:
            protected_values = []
        elif isinstance(raw_protected_values, list):
            protected_values = raw_protected_values
        else:
            # Si viene un solo valor (ej: 1 o "female"), lo convertimos a lista
            protected_values = [raw_protected_values]
            
        favorable_outcomes = get_val(fairness_section, "favorable_outcomes", [])
        if favorable_outcomes is None: 
            favorable_outcomes = [] # Asegurar que no sea None si el json tenía null
        
        target_column = get_val(general_section, "target_column", "")
        # if not target_column:
        #      target_column = fairness_section.get("target_column", "")

        if not protected_feature or not target_column:
            raise ValueError("Configuración incompleta: falta protected_feature o target_column")

        return protected_feature, protected_values, target_column, favorable_outcomes
    except Exception as e:
        raise ValueError(f"Configuración incompleta. Faltan datos obligatorios.\n"
                                    f" - Protected Feature: '{protected_feature}'\n"
                                    f" - Target Column: '{target_column}'")

def convert_to_aif360_dataset(df, target_column, protected_feature, protected_values, favorable_outcomes):
    df_aif = df.copy()
    
    # Mapeo de seguridad para label favorable
    fav_label = favorable_outcomes[0] if len(favorable_outcomes) > 0 else 1
    unfav_label = 0 if fav_label == 1 else 1

    # Crear dataset AIF360
    dataset = BinaryLabelDataset(
        df=df_aif,
        label_names=[target_column],
        protected_attribute_names=[protected_feature],
        favorable_label=fav_label,
        unfavorable_label=unfav_label
    )
    
    # Definición de grupos robusta
    # Nota: AIF360 es estricto con los tipos. Si en CSV es float (0.0) y en config es int (0), falla.
    # Convertimos todo a string para comparar o usamos isclose si fuera necesario, 
    # pero aquí confiamos en la carga correcta de pandas.
    unique_vals = df_aif[protected_feature].unique()
    
    privileged_groups = [{protected_feature: v} for v in unique_vals if v not in protected_values]
    unprivileged_groups = [{protected_feature: v} for v in protected_values]

    return dataset, privileged_groups, unprivileged_groups

def get_aif360_metrics(test_data, y_pred, factsheet):
    protected_feature, protected_values, target_column, favorable_outcomes = load_fairness_config(factsheet)

    dataset_true, priv, unpriv = convert_to_aif360_dataset(
        test_data, target_column, protected_feature, protected_values, favorable_outcomes
    )
    dataset_pred = dataset_true.copy()
    dataset_pred.labels = y_pred.reshape(-1, 1)
    
    return ClassificationMetric(dataset_true, dataset_pred, unprivileged_groups=unpriv, privileged_groups=priv)

def compute_holistic_metrics(context, group_col="Group", target_col="Target"):
    df = context.test_data.copy()
    return {
        'group': classification_bias_metrics(
            group_a = df[group_col] == 1,
            group_b = df[group_col] == 0,
            y_true = df[target_col],
            y_pred = context.y_pred_test,
            metric_type='group'
        ),
        'individual': classification_bias_metrics(
            group_a = df[group_col] == 1,
            group_b = df[group_col] == 0,
            y_true = df[target_col],
            y_pred = context.y_pred_test,
            metric_type='individual'
        )
    }


# === MAIN ANALYSE ===

def analyse(context, config): 
    # Helper seguro para extraer thresholds
    def get_thresh(key):
        val = config.get(key, {}).get("thresholds", {}).get("value")
        if val is None:
            # Fallback seguro si falta la config
            print(f"Warning: Missing thresholds for {key} in config. Using default [0.1, 0.2, 0.3, 0.4].")
            return [0.1, 0.2, 0.3, 0.4] 
        return val
    
    # Calculamos
    output = {
        "underfitting": underfitting_score(context, get_thresh("score_underfitting")),
        "overfitting": overfitting_score(context, get_thresh("score_overfitting")),
        "class_balance": class_balance_score(context, None),
        "statistical_parity_difference": stat_parity_score(context, get_thresh("score_statistical_parity_difference")),
        "disparate_impact": disparate_impact_score(context, get_thresh("score_disparate_impact")),
        "equal_opportunity_difference": eq_opp_score(context, get_thresh("score_equal_opportunity_difference")),
        "average_odds_difference": avg_odds_score(context, get_thresh("score_average_odds_difference")),

        "accuracy_parity": accuracy_parity_score(context, get_thresh("score_accuracy_parity")),
        "predictive_parity": predictive_parity_score(context, get_thresh("score_predictive_parity")),
        "treatment_equality": treatment_equality_score(context, get_thresh("score_treatment_equality")),
        "calibration": calibration_score(context, get_thresh("score_calibration"), 10),
        "well_calibration": well_calibration_score(context, get_thresh("score_well_calibration"), 10),

        "generalized_entropy": generalized_entropy_score(context, get_thresh("score_generalized_entropy"), alpha=2),
        "theil_index": theil_index_score(context, get_thresh("score_theil_index")),
        "coefficient_variation": coefficient_variation_score(context, get_thresh("score_coefficient_variation")),
        "consistency": consistency_score(context, get_thresh("score_consistency")),
        "class_imbalance": class_imbalance_score(context, get_thresh("score_class_imbalance")),
        "kl_divergence": kl_divergence_score(context, get_thresh("score_kl_divergence")),
        # "conditional_dp": conditional_dp_score(model, test_data, factsheet, th_conditional_dp),
        "smoothed_edf": smoothed_edf_score(context, get_thresh("score_smoothed_edf")),
        "bias_amplification": bias_amplification_score(context, get_thresh("score_bias_amplification")), #REV
        #"between_group_ge": between_group_ge_score(model, test_data, factsheet, th_between_group_ge),
        "cohens_d": cohens_d_score(context, get_thresh("score_cohens_d")),
        # "two_sd_rule": two_sd_rule_score(model, test_data, factsheet)
    }
    
    scores = {k: v.score for k, v in output.items()}
    properties = {k: v.properties for k, v in output.items()}

    # Holistic
    # Imprimirmos las dos métricas holísticas de HolisticAI, aunque no las incluimos en el score global
    holistic_metrics = compute_holistic_metrics(context)
    print("HolisticAI Group Fairness Metrics:", holistic_metrics['group'])
    print("HolisticAI Individual Fairness Metrics:", holistic_metrics['individual'])

    return Result(score=scores, properties=properties)


# === METRIC FUNCTIONS ===

def underfitting_score(context, thresholds):
    try:
        acc = metrics.accuracy_score(context.y_test, context.y_pred_test)
        score = calculate_score(acc, thresholds)

        props = {
            "Metric Description": "Compares the models achieved test accuracy against a baseline.",
            "Depends on": "Model, Test Data",
            "Test Accuracy": f"{acc:.2%}",
            "Conclusion": (
                "Model mildly underfitting" if score <= 3
                else "Model well fitted"
            )
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def overfitting_score(context, thresholds):
    try:
        # 1. Verificar primero si hay underfitting (Test Acc > 0.6 o un umbral razonable)
        test_acc = metrics.accuracy_score(context.y_test, context.y_pred_test)
        train_acc = metrics.accuracy_score(context.y_train, context.y_pred_train)
        
        if test_acc < 0.6: # Si el modelo es muy malo, no medimos overfitting
             return Result(np.nan, {"Info": "Test accuracy too low (<60%) to measure overfitting"})

        diff = train_acc - test_acc
        score = calculate_score(diff, thresholds)
        
        props = {
            "Metric Description": (
                "Overfitting is present if the training accuracy is significantly higher than the test accuracy"
            ),
            "Depends on": "Model, Training Data, Test Data",
            "Training Accuracy": f"{train_acc:.2%}",
            "Test Accuracy": f"{test_acc:.2%}",
            "Train Test Accuracy Difference": f"{diff:.2%}",
            "Conclusion": (
                "Model is overfitting" if diff > 0.05
                else "Model is not overfitting"
            )
        }

        return Result(score, props)
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})

def class_balance_score(context, thresholds=None):
    try:
        _, _, target, _ = load_fairness_config(context.factsheet)
        counts = context.train_data[target].value_counts().sort_index().to_numpy()
        p_val = chisquare(counts).pvalue
        score = 5 if p_val >= 0.05 else 1

        props = {
            "Metric Description": "Measures how well the training data is balanced or unbalanced",
            "Depends on": "Training Data",
            "P-Value": f"{p_val:.4f}",
            "Conclusion": (
                "Classes are balanced" if p_val >= 0.05
                else "Classes are imbalanced"
            )
        }

        return Result(score, props)
    except Exception as e:
        return Result(np.nan, {"Error": str(e)})
    

def stat_parity_score(context, thresholds): # IMPORANTE: TEST
    try:
        # Cargar configuración de fairness
        prot, vals, target, fav = load_fairness_config(context.factsheet)

        #Añadimos al test_data una columna con las predicciones del modelo
        dataset = context.test_data.copy()
        dataset[target] = context.y_pred_test

        ds, priv, unpriv = convert_to_aif360_dataset(dataset, target, prot, vals, fav)

        # Crear métrica
        m = BinaryLabelDatasetMetric(ds, unprivileged_groups=unpriv, privileged_groups=priv)
        
        # Convertimos a array para contar manualmente
        labels = ds.labels.flatten()  # y_true
        protected_attr = ds.protected_attributes[:, 0]  # atributo protegido (0 o 1)

        count_prot_total = (protected_attr == 1).sum()
        count_prot_fav  = ((protected_attr == 1) & (labels == 1)).sum()
        count_unprot_total = (protected_attr == 0).sum()
        count_unprot_fav  = ((protected_attr == 0) & (labels == 1)).sum()

        ratio_prot = count_prot_fav / count_prot_total * 100
        ratio_unprot = count_unprot_fav / count_unprot_total * 100
        
        val = ratio_prot/100 - ratio_unprot/100  # Esto es lo que devuelve AIF360
        
        score = calculate_score(val, thresholds)

        props = {
        "Metric Description": (
            "The spread between the percentage of observations from the majority group receiving a favorable outcome compared to the protected group."
        ),
        "Depends on": (
            "Training Data, Factsheet (Definition of Protected Group and Favorable Outcome)"
        ),
        "|{{x|x is protected, y_pred is favorable}}|": f"{count_prot_fav}",
        "|{{x|x is protected}}|": f"{count_prot_total}",
        "Favored Minority (Protected Group) Ratio": f"P(y_hat=favorable|protected=True) = {ratio_prot:.2f}%",
        "|{{x|x is not protected, y_pred is favorable}}|": f"{count_unprot_fav}",
        "|{{x|x is not protected}}|": f"{count_unprot_total}",
        "Favored Majority (Unprotected Group) Ratio": f"P(y_hat=favorable|protected=False) = {ratio_unprot:.2f}%",
        "Formula": (
            "Favored Majority Ratio - Favored Minority Ratio"
        ),
        "Statistical Parity Difference": f"{val*100:.2f}%",
        "Statistical Parity Difference AIF360": f"{m.statistical_parity_difference():.4f}"
        }

        return Result(score, props)
    
    except Exception as e:
        print(str(e))
        return Result(np.nan, {"Error": str(e)})


def disparate_impact_score(context, thresholds):
    try:
        prot, vals, target, fav = load_fairness_config(context.factsheet)

        #Añadimos al test_data una columna con las predicciones del modelo
        dataset = context.test_data.copy()
        dataset[target] = context.y_pred_test

        ds, priv, unpriv = convert_to_aif360_dataset(dataset, target, prot, vals, fav)

        # Métrica AIF360
        m = BinaryLabelDatasetMetric(ds,unprivileged_groups=unpriv,privileged_groups=priv)

        # Datos manuales
        labels = ds.labels.flatten()
        protected_attr = ds.protected_attributes[:, 0]

        # Conteos
        count_prot_total = (protected_attr == 1).sum()
        count_prot_fav = ((protected_attr == 1) & (labels == 1)).sum()

        count_unprot_total = (protected_attr == 0).sum()
        count_unprot_fav = ((protected_attr == 0) & (labels == 1)).sum()

        # Probabilidades
        p_prot = count_prot_fav / count_prot_total
        p_unprot = count_unprot_fav / count_unprot_total

        # Disparate Impact
        val = p_prot / p_unprot

        score = calculate_score(val, thresholds)

        props = {
        "Metric Description": (
            "Is quotient of the ratio of samples from the protected group receiving a favorable prediction divided by the ratio of samples from the unprotected group"
        ),
        "Depends on": (
            "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)"
        ),
        "|{{x|x is protected, y_pred is favorable}}|": f"{count_prot_fav}",
        "|{{x|x is protected}}|": f"{count_prot_total}",
        "Protected Favored Ratio": f"P(y_hat=favorable|protected=True) = {p_prot*100:.2f}%",
        "|{{x|x is unprotected, y_pred is favorable}}|": f"{count_unprot_fav}",
        "|{{x|x is unprotected}}|": f"{count_unprot_total}",
        "Unprotected Favored Ratio": f"P(y_hat=favorable|protected=False) = {p_unprot*100:.2f}%",
        "Formula": (
            "Disparate Impact = Protected Favored Ratio / Unprotected Favored Ratio"
        ),
        "Disparate Impact": f"{val:.2f}",
        "Disparate Impact AIF360": f"{m.disparate_impact():.4f}"
        }

        return Result(score, props)

    except Exception as e:
        print(str(e))
        return Result(np.nan, {"Error": str(e)})


def eq_opp_score(context, thresholds):
    try:
        m = get_aif360_metrics(context.test_data, context.y_pred_test, context.factsheet)

        # # Verdaderos positivos
        # tp_prot = ((protected_attr == 1) & (y_true == 1) & (y_pred == 1)).sum()
        # tp_unprot = ((protected_attr == 0) & (y_true == 1) & (y_pred == 1)).sum()

        # # Positivos reales
        # pos_prot = ((protected_attr == 1) & (y_true == 1)).sum()
        # pos_unprot = ((protected_attr == 0) & (y_true == 1)).sum()

        # tpr_prot = tp_prot / pos_prot if pos_prot > 0 else 0.0
        # tpr_unprot = tp_unprot / pos_unprot if pos_unprot > 0 else 0.0

        tpr_unprot = m.true_positive_rate(privileged=True)
        tpr_prot   = m.true_positive_rate(privileged=False)

        val = tpr_prot - tpr_unprot

        score = calculate_score(val, thresholds)

        props =  {
        "Metric Description": (
            "Difference in true positive rates between protected and unprotected group."
        ),
        "Depends on": (
            "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)"
        ),
        #"TP Unprotected Group": f"{tp_unprot}",
        #"Positive Unprotected Group": f"{pos_unprot}",
        "TPR Unprotected Group": f"{tpr_unprot:.2%}",
        #"TP Protected Group": f"{tp_prot}",
        #"Positive Protected Group": f"{pos_prot}",
        "TPR Protected Group": f"{tpr_prot:.2%}",
        "Formula": (
            "Equal Opportunity Difference = TPR Protected Group - TPR Unprotected Group"
        ),
        "Equal Opportunity Difference": f"{val*100:.2f}%",
        "Equal Opportunity Difference AIF360": f"{m.equal_opportunity_difference():.4f}"
        }
        return Result(score, props)
    except Exception as e:
        print(str(e))
        return Result(np.nan, {"Error": str(e)})

def avg_odds_score(context, thresholds):
    try:
        m = get_aif360_metrics(context.test_data, context.y_pred_test, context.factsheet)

        tpr_unprot = m.true_positive_rate(privileged=True)
        tpr_prot   = m.true_positive_rate(privileged=False)
        fpr_unprot = m.false_positive_rate(privileged=True)
        fpr_prot   = m.false_positive_rate(privileged=False)

        # # TPR
        # tp_prot = ((protected_attr == 1) & (y_true == 1) & (y_pred == 1)).sum()
        # tp_unprot = ((protected_attr == 0) & (y_true == 1) & (y_pred == 1)).sum()
        # pos_prot = ((protected_attr == 1) & (y_true == 1)).sum()
        # pos_unprot = ((protected_attr == 0) & (y_true == 1)).sum()

        # tpr_prot = tp_prot / pos_prot if pos_prot > 0 else 0.0
        # tpr_unprot = tp_unprot / pos_unprot if pos_unprot > 0 else 0.0

        # # FPR
        # fp_prot = ((protected_attr == 1) & (y_true == 0) & (y_pred == 1)).sum()
        # fp_unprot = ((protected_attr == 0) & (y_true == 0) & (y_pred == 1)).sum()
        # neg_prot = ((protected_attr == 1) & (y_true == 0)).sum()
        # neg_unprot = ((protected_attr == 0) & (y_true == 0)).sum()

        # fpr_prot = fp_prot / neg_prot if neg_prot > 0 else 0.0
        # fpr_unprot = fp_unprot / neg_unprot if neg_unprot > 0 else 0.0

        val = 0.5 * ((tpr_prot - tpr_unprot) + (fpr_prot - fpr_unprot))
        
        score = calculate_score(val, thresholds)

        props =  {
        "Metric Description": (
            "Is the average of difference in false positive rates and true positive rates between the protected and unprotected group"
        ),
        "Depends on": (
            "Model, Test Data, Factsheet (Definition of Protected Group and Favorable Outcome)"
        ),
        "FPR Unprotected Group": f"{fpr_unprot:.2%}",
        "FPR Protected Group": f"{fpr_prot:.2%}",
        "TPR Unprotected Group": f"{tpr_unprot:.2%}",
        "TPR Protected Group": f"{tpr_prot:.2%}",
        "Formula": (
            "0.5*(TPR Protected - TPR Unprotected) + 0.5*(FPR Protected - FPR Unprotected)"
        ),
        "Average Odds Difference": f"{val*100:.2f}%",
        "Average Odds Difference AIF360": f"{m.average_odds_difference():.4f}"
        }

        return Result(score, props)

    except Exception as e:
        print(str(e))
        return Result(np.nan, {"Error": str(e)})



def accuracy_parity_score(context, thresholds):
    try:
        m = get_aif360_metrics(context.test_data, context.y_pred_test, context.factsheet)

        acc_unprot = m.accuracy(privileged=True)
        acc_prot   = m.accuracy(privileged=False)

        val = acc_prot - acc_unprot
        score = calculate_score(val, thresholds)

        props = {
            "Metric Description": (
                "Measures whether prediction accuracy is equal across protected and unprotected groups."
            ),
            "Depends on": "Model, Test Data, Factsheet",
            "Accuracy Unprotected Group": f"{acc_unprot:.2%}",
            "Accuracy Protected Group": f"{acc_prot:.2%}",
            "Formula": (
                "Accuracy Parity = Accuracy Protected - Accuracy Unprotected"
            ),
            "Accuracy Parity Difference": f"{val*100:.2f}%"
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def predictive_parity_score(context, thresholds):
    try:
        m = get_aif360_metrics(context.test_data, context.y_pred_test, context.factsheet)

        ppv_unprot = m.positive_predictive_value(privileged=True)
        ppv_prot   = m.positive_predictive_value(privileged=False)

        npv_unprot = m.negative_predictive_value(privileged=True)
        npv_prot   = m.negative_predictive_value(privileged=False)

        val_ppv = ppv_prot - ppv_unprot
        val_npv = npv_prot - npv_unprot

        val = 0.5 * (val_ppv + val_npv)
        score = calculate_score(val, thresholds)

        props = {
            "Metric Description": (
                "Checks equality of PPV and NPV between protected and unprotected groups."
            ),
            "Depends on": "Model, Test Data, Factsheet",
            "PPV Unprotected": f"{ppv_unprot:.2%}",
            "PPV Protected": f"{ppv_prot:.2%}",
            "NPV Unprotected": f"{npv_unprot:.2%}",
            "NPV Protected": f"{npv_prot:.2%}",
            "Formula": (
                "0.5 * (PPV Protected - PPV Unprotected) + "
                "0.5 * (NPV Protected - NPV Unprotected)"
            ),
            "Predictive Parity Difference": f"{val*100:.2f}%"
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def treatment_equality_score(context, thresholds):
    try:
        m = get_aif360_metrics(context.test_data, context.y_pred_test, context.factsheet)

        fn_unprot = m.num_false_negatives(privileged=True)
        fp_unprot = m.num_false_positives(privileged=True)

        fn_prot = m.num_false_negatives(privileged=False)
        fp_prot = m.num_false_positives(privileged=False)

        ratio_unprot = fn_unprot / fp_unprot if fp_unprot > 0 else np.inf
        ratio_prot   = fn_prot / fp_prot if fp_prot > 0 else np.inf

        val = ratio_prot - ratio_unprot
        score = calculate_score(val, thresholds)

        props = {
            "Metric Description": (
                "Measures equality of the ratio between false negatives and false positives across groups."
            ),
            "Depends on": "Model, Test Data, Factsheet",
            "FN Unprotected": fn_unprot,
            "FP Unprotected": fp_unprot,
            "FN Protected": fn_prot,
            "FP Protected": fp_prot,
            "FN/FP Unprotected": f"{ratio_unprot:.4f}",
            "FN/FP Protected": f"{ratio_prot:.4f}",
            "Formula": (
                "(FN/FP)_Protected - (FN/FP)_Unprotected"
            ),
            "Treatment Equality Difference": f"{val:.4f}"
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def calibration_score(context, thresholds, n_bins=10):
    try:
        if not hasattr(context.model, "predict_proba"):
            return Result(np.nan, {
                "Info": "Model does not provide probabilistic scores (predict_proba required)"
            })

        prot, vals, target, _ = load_fairness_config(context.factsheet)

        df_test = context.test_data.copy()
        df_test["score"] = context.y_prob_test[:, 1]  # Solo la clase positiva
        df_test["y"] = context.y_test

        df_test["bin"] = pd.qcut(df_test["score"], n_bins, duplicates="drop")

        cal = (
            df_test.groupby([prot, "bin"])
              .apply(lambda g: g["y"].mean())
              .unstack(level=0)
        )

        diff = (cal.iloc[:, 0] - cal.iloc[:, 1]).abs().mean()
        score = calculate_score(diff, thresholds)

        props = {
            "Metric Description": (
                "Checks whether individuals with the same predicted score have the same outcome probability across groups. If the model is fair, in a bin where the probability is 0.7, both groups should have approximately 70\% actual positive cases."
            ),
            "Depends on": "Model, Test Data, Probabilistic Scores",
            "Mean Calibration Gap": f"{diff:.4f}",
            "Bins Used": n_bins,
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def well_calibration_score(context, thresholds, n_bins=10):
    try:
        if not hasattr(context.model, "predict_proba"):
            return Result(np.nan, {
                "Info": "Model does not provide probabilistic scores (predict_proba required)"
            })

        prot, vals, target, _ = load_fairness_config(context.factsheet)

        df_test = context.test_data.copy()
        df_test["score"] = context.y_prob_test[:, 1]  # Solo la clase positiva
        df_test["y"] = context.test_data[target].values
        df_test["bin"] = pd.qcut(df_test["score"], n_bins, duplicates="drop")

        err = (
            df_test.groupby("bin")
              .apply(lambda g: abs(g["y"].mean() - g["score"].mean()))
              .mean()
        )

        score = calculate_score(err, thresholds)

        props = {
            "Metric Description": (
                "Checks whether predicted probabilities match empirical outcome frequencies."
            ),
            "Depends on": "Model, Test Data, Probabilistic Scores",
            "Mean Calibration Error": f"{err:.4f}",
            "Bins Used": n_bins,
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def generalized_entropy_score(context, thresholds, alpha=2):
    try:
        m = get_aif360_metrics(context.test_data, context.y_pred_test, context.factsheet)

        val = m.generalized_entropy_index(alpha=alpha)
        score = calculate_score(val, thresholds)

        props = {
            "Metric Description": (
                "Measures inequality in the distribution of benefits (correct predictions)."
            ),
            "Depends on": "Model, Test Data",
            "Alpha": alpha,
            "Benefit Definition": "b_i = 1 if prediction is correct, 0 otherwise",
            "Generalized Entropy (AIF360)": f"{val:.6f}",
            "Conclusion": (
                "Perfect equality" if val == 0
                else "Prediction benefits unequally distributed"
            )
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})

def theil_index_score(context, thresholds):
    return generalized_entropy_score(
        context, thresholds, alpha=1
    )

def coefficient_variation_score(context, thresholds):
    try:
        m = get_aif360_metrics(context.test_data, context.y_pred_test, context.factsheet)
        val = np.sqrt(2*m.generalized_entropy_index(alpha=2))
        score = calculate_score(val, thresholds)

        props = {
            "Metric Description": (
                "Measures dispersion (instability) in prediction benefits."
            ),
            "Depends on": "Model, Test Data",
            "Coefficient of Variation": f"{val:.6f}",
            "Coefficient of Variation (AIF360)": f"{m.coefficient_of_variation():.6f}",
            "Formula": "CV = sigma / mu"
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


from sklearn.neighbors import NearestNeighbors

def consistency_score(context, thresholds, k=5):
    try:
        m = get_aif360_metrics(context.test_data, context.y_pred_test, context.factsheet)
        prot, _, target, _ = load_fairness_config(context.factsheet)
        X_test = context.X_test.copy()

        nn = NearestNeighbors(n_neighbors=k+1).fit(X_test)
        _, idx = nn.kneighbors(X_test)

        diffs = []
        for i in range(len(X_test)):
            neigh = idx[i][1:]
            diffs.append(abs(context.y_pred_test[i] - np.mean(context.y_pred_test[neigh])))

        val = 1 - np.mean(diffs)
        score = calculate_score(val, thresholds)

        cons_aif = m.consistency()
        if isinstance(cons_aif, np.ndarray):
            cons_aif = cons_aif.mean()

        props = {
            "Metric Description": (
                "Measures whether similar individuals receive similar predictions."
            ),
            "Depends on": "Model, Test Data",
            "k Neighbors": k,
            "Consistency Score": f"{val:.4f}",
            "Consistency Score (AIF360)": f"{cons_aif:.4f}",
            "Formula": "Consistency = 1 - average(|y_hat_i - mean(y_hat_neighbors)|)"
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


from aif360.sklearn.metrics import class_imbalance

def class_imbalance_score(context, thresholds):
    try:
        dataset = context.test_data
        factsheet = context.factsheet
        prot, vals, target, fav = load_fairness_config(factsheet)

        # Favorable label (por defecto 1 si no se indica)
        fav_label = fav[0] if fav else 1

        # Máscara del grupo protegido
        prot_mask = dataset[prot].isin(vals)

        # Cálculo manual
        # Np = ((prot_mask) & (dataset[target] == fav_label)).sum()
        # Nu = ((~prot_mask) & (dataset[target] == fav_label)).sum()  # ¡corregido a fav_label, no != fav_label!
        # val_manual = (Nu - Np) / (Nu + Np)
        Np_total = prot_mask.sum()
        Nu_total = (~prot_mask).sum()
        val_manual = (Nu_total - Np_total) / (Nu_total + Np_total)

        # Score simple
        score = 5 if abs(val_manual) < 0.1 else 1

        # Cálculo usando AIF360 sklearn
        y_true = dataset[target]
        prot_attr = dataset[prot]

        # Grupo privilegiado: el que no está en los valores protegidos
        priv_group = [v for v in prot_attr.unique() if v not in vals][0]

        val_aif360 = class_imbalance(
            y_true=y_true,
            prot_attr=prot_attr,
            priv_group=priv_group
        )

        props = {
            "Metric Description": "Measures imbalance between privileged and unprivileged samples.",
            "Depends on": "Dataset",
            "CI (manual)": f"{val_manual:.4f}",
            "CI (AIF360)": f"{val_aif360:.4f}",
            "Interpretation": "0 indicates perfect balance"
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})



from scipy.stats import entropy

def kl_divergence_score(context, thresholds):
    try:
        #m = get_aif360_metrics(model, test_data, factsheet)
        prot, vals, target, _ = load_fairness_config(context.factsheet)

        y = context.test_data[target]
        prot_mask = context.test_data[prot].isin(vals)

        Pp = y[~prot_mask].value_counts(normalize=True).sort_index()
        Pu = y[prot_mask].value_counts(normalize=True).sort_index()

        Pp, Pu = Pp.align(Pu, fill_value=1e-9)

        val = entropy(Pp, Pu)
        score = calculate_score(val, thresholds)

        props = {
            "Metric Description": (
                "Measures divergence between label distributions across groups."
            ),
            "Depends on": "Dataset",
            "KL Divergence": f"{val:.6f}",
            #"KL Divergence (AIF360)": f"{m.kl_divergence():.6f}"
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


# def conditional_dp_score(dataset, factsheet, thresholds, conditioning_cols):
#     try:
#         m = get_aif360_metrics(None, dataset, factsheet)
#         prot, vals, target, fav = load_fairness_config(factsheet)
#         fav_label = fav[0] if fav else 1

#         total = 0
#         weighted = 0

#         for _, g in dataset.groupby(conditioning_cols):
#             Ni = len(g)
#             if Ni == 0:
#                 continue

#             p_pos = (g[target] == fav_label).mean()
#             p_neg = 1 - p_pos

#             weighted += Ni * (p_neg - p_pos)
#             total += Ni

#         val = weighted / total
#         score = calculate_score(val, thresholds)

#         props = {
#             "Metric Description": (
#                 "Measures demographic parity conditioned on additional variables."
#             ),
#             "Depends on": "Dataset",
#             "CDD": f"{val:.4f}",
#             "CDD (AIF360)": f"{m.conditional_demographic_parity(conditioning_cols):.4f}"
#         }

#         return Result(score, props)

#     except Exception as e:
#         return Result(np.nan, {"Error": str(e)})

def smoothed_edf_score(context, thresholds, alpha=1.0, epsilon=0.1):
    try:
        m = get_aif360_metrics(context.test_data, context.y_pred_test, context.factsheet)
        if not hasattr(context.model, "predict_proba"):
            return Result(np.nan, {"Info": "predict_proba required"})

        prot, vals, target, _ = load_fairness_config(context.factsheet)
        X = context.test_data.drop(target, axis=1)
        probs = context.model.predict_proba(X)[:, 1]

        df = context.test_data.copy()
        df["p"] = probs

        groups = df.groupby(prot)["p"].mean() + alpha
        ratios = groups.max() / groups.min()

        val = abs(np.log(ratios))
        score = calculate_score(val, thresholds)

        props = {
            "Metric Description": "Smoothed Equality of Distributions Fairness",
            "Depends on": "Model, Probabilities",
            "Alpha": alpha,
            "Epsilon": epsilon,
            "EDF Log-Ratio": f"{val:.4f}",
            "EDF (AIF360)": f"{m.smoothed_empirical_differential_fairness(concentration=epsilon):.4f}"
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})

def bias_amplification_score(context, thresholds):
    try:
        #m = get_aif360_metrics(context.test_data, context.y_pred_test, context.factsheet)
        prot, vals, target, _ = load_fairness_config(context.factsheet)

        y = context.test_data[target]
        y_hat = context.model.predict(context.test_data.drop(target, axis=1))

        bias_y = abs(y[context.test_data[prot].isin(vals)].mean() - y[~context.test_data[prot].isin(vals)].mean())
        bias_yhat = abs(y_hat[context.test_data[prot].isin(vals)].mean() - y_hat[~context.test_data[prot].isin(vals)].mean())

        val = bias_yhat - bias_y
        score = calculate_score(val, thresholds)

        props = {
            "Metric Description": "Measures whether model amplifies dataset bias.",
            "Bias Dataset": f"{bias_y:.4f}",
            "Bias Predictions": f"{bias_yhat:.4f}",
            "Bias Amplification": f"{val:.4f}",
            #"Bias Amplification (AIF360)": f"{m.bias_amplification():.4f}"
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})

# def between_group_ge_score(model, test_data, factsheet, thresholds, alpha=1):
#     try:
#         m = get_aif360_metrics(model, test_data, factsheet)
#         val = m.between_group_generalized_entropy_index(alpha=alpha)
#         score = calculate_score(val, thresholds)

#         props = {
#             "Metric Description": (
#                 "Measures inequality strictly between protected groups."
#             ),
#             "Alpha": alpha,
#             "Between-group GE (AIF360)": f"{val:.6f}"
#         }

#         return Result(score, props)

#     except Exception as e:
#         return Result(np.nan, {"Error": str(e)})


def cohens_d_score(context, thresholds):
    try:
        prot, vals, target, _ = load_fairness_config(context.factsheet)
        y_hat = context.model.predict(context.test_data.drop(target, axis=1))

        g1 = y_hat[context.test_data[prot].isin(vals)]
        g2 = y_hat[~context.test_data[prot].isin(vals)]

        mu1, mu2 = g1.mean(), g2.mean()
        sigma = np.sqrt((g1.var() + g2.var()) / 2)

        val = (mu1 - mu2) / sigma if sigma > 0 else 0.0
        score = calculate_score(val, thresholds)

        props = {
            "Metric Description": "Effect size between protected groups.",
            "Cohen's D": f"{val:.4f}",
            "Formula": "(mu1 - mu2) / sigma_pooled"
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})


def two_sd_rule_score(context):
    try:
        prot, vals, target, _ = load_fairness_config(context.factsheet)
        y_hat = context.model.predict(context.test_data.drop(target, axis=1))

        g1 = y_hat[context.test_data[prot].isin(vals)]
        g2 = y_hat[~context.test_data[prot].isin(vals)]

        mu_diff = abs(g1.mean() - g2.mean())
        sigma = np.std(y_hat)

        violated = mu_diff > 2 * sigma
        score = 1 if violated else 5

        props = {
            "Metric Description": "Heuristic adverse impact detection rule.",
            "Mean Difference": f"{mu_diff:.4f}",
            "2sigma Threshold": f"{2*sigma:.4f}",
            "Violated": violated
        }

        return Result(score, props)

    except Exception as e:
        return Result(np.nan, {"Error": str(e)})
