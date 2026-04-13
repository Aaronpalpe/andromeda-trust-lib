from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from holisticai.security.metrics import (
    attribute_attack_score,
    data_minimization_score,
    privacy_risk_score,
    shapr_score,
)
from holisticai.security.metrics import k_anonymity as hol_k_anonymity
from holisticai.security.metrics import l_diversity as hol_l_diversity

# =============================================================================
# Epsilon DP Leakage
# =============================================================================
def epsilon_dp(epsilon: float) -> Dict[str, float]:
    """
    Compute a score for epsilon DP leakage based on predefined thresholds.

    Parameters
    ----------
    epsilon: float
        The epsilon value to evaluate for differential privacy leakage.

    Returns
    -------
    float: The epsilon value provided as input, indicating the level of privacy leakage, where lower values suggest stronger privacy guarantees.
    """
    if epsilon is None:
        raise ValueError("Epsilon value is required for epsilon_dp metric. Please provide it in the factsheet.")
    
    return {
        "value": epsilon,
    }

# =============================================================================
# Helper: Log-loss per instance
# =============================================================================

def _calculate_losses(model, X, y) -> np.ndarray:
    """
    Compute Log Loss (Cross-Entropy) for each instance.
    """
    probs = model.predict_proba(X)

    if hasattr(model, "classes_"):
        class_map = {c: i for i, c in enumerate(model.classes_)}
        y_idx = np.array([class_map[label] for label in y])
    else:
        y_idx = y.astype(int)

    true_probs = probs[np.arange(len(y)), y_idx]
    true_probs = np.clip(true_probs, 1e-15, 1 - 1e-15)

    return -np.log(true_probs)


# =============================================================================
# Epsilon Star
# =============================================================================

def epsilon_star(
    model,
    X_train : pd.DataFrame,
    y_train : pd.Series,
    X_test : pd.DataFrame,
    y_test : pd.Series,
) -> Dict[str, float]:
    """
    Compute empirical epsilon* based on Loss Distribution as Definition 2 of the paper.
    It first selects a subset of training and test data and computes the loss of the model on each sample. 
    Then, it constructs a score vector by concatenating the negative losses of training and test samples,
    and defines corresponding labels (1 for training, 0 for test). Using these scores, it calculates 
    the false positive rate (FPR) and false negative rate (FNR) across thresholds, applies a small 
    smoothing factor delta to avoid division by zero, and computes four cocycles comparing training and test error rates. 
    Finally, it returns the logarithm of the maximal ratio, which serves as epsilon_star_val, 
    a numeric measure of the model's memorization risk or susceptibility to membership inference.

    Parameters
    ----------
    model : object
        The trained model to evaluate.
    X_train : pd.DataFrame
        The training features.
    y_train : pd.Series
        The training labels.
    X_test : pd.DataFrame
        The test features.
    y_test : pd.Series
        The test labels.

    Returns
    -------
    Dict[str, float]: The computed epsilon* value, representing the empirical privacy leakage of the model, where lower values indicate stronger privacy guarantees.
    """
    if getattr(model, "_is_regressor", False) or (hasattr(y_train, "dtype") and y_train.dtype.kind == 'f'):
        raise ValueError("Not suitable for regression models or continuous targets. Epsilon* is designed for classification tasks with discrete labels. It uses roc_curve which requires binary labels.")
    
    idx = np.random.choice(len(X_train), min(5000, len(X_train)), replace=False)
    X_train_small = X_train.iloc[idx]
    y_train_small = y_train.iloc[idx] if hasattr(y_train, "iloc") else y_train[idx]

    #idx =  np.random.choice(min(5000, len(X_test)), replace=False) 
    idx = np.arange(min(5000, len(X_test)))

    X_test_small = X_test.iloc[idx]
    y_test_small = y_test.iloc[idx] if hasattr(y_test, "iloc") else y_test[idx]

    loss_train = _calculate_losses(model, X_train_small, y_train_small)
    loss_test  = _calculate_losses(model, X_test_small, y_test_small)

    scores = np.concatenate([-loss_train, -loss_test])
    y_true = np.concatenate([np.ones(len(loss_train)), np.zeros(len(loss_test))])

    fpr, tpr, _ = roc_curve(y_true, scores)

    fpr = np.clip(fpr, 1e-10, 1 - 1e-10)
    tpr = np.clip(tpr, 1e-10, 1 - 1e-10)
    fnr = 1 - tpr

    delta = 1.0 / len(loss_train) if len(loss_train) > 0 else 1e-5

    m1 = (1 - delta - fnr) / fpr
    m2 = (1 - delta - fpr) / fnr
    m3 = (fnr - delta) / (1 - fpr)
    m4 = (fpr - delta) / (1 - fnr)

    epsilon_star_val = np.log(
        np.nanmax(np.maximum.reduce([m1, m2, m3, m4, np.ones_like(m1)]))
    )

    if np.isnan(epsilon_star_val) or np.isinf(epsilon_star_val):
        raise ValueError("Epsilon* calculation resulted in NaN or Inf. Check if your model and data are suitable for this metric.")

    return {
        "value": float(epsilon_star_val),
        "delta": float(delta),
    }


# =============================================================================
# SHAPr
# =============================================================================

def shapr(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    random_state: int = 42,
) -> Dict[str, float]:
    '''
    This function estimates the SHAPr score, which quantifies the risk of membership inference attacks 
    by analyzing how test samples relate to the training data in the prediction space. Specifically, 
    it fits a k-nearest neighbors (kNN) model using the predictions on the training set as features and 
    their true labels as targets. Then, for each test sample, it identifies its nearest neighbors among 
    the training samples based on their predicted values.

    The method evaluates whether these neighboring training samples share the same true label as the test sample, 
    capturing how strongly the prediction of the test point is associated with a specific class in the training data. 
    The resulting score reflects the degree to which a test sample is surrounded by training samples of the same class 
    in the prediction space. Higher scores indicate that the test sample lies in a region dominated by a single class, 
    suggesting a greater risk of privacy leakage.

    Parameters
    ----------
    model : object
        The trained model to evaluate.
    X_train : pd.DataFrame
        The training features.
    y_train : pd.Series
        The training labels.
    X_test : pd.DataFrame
        The test features.
    y_test : pd.Series
        The test labels.
    random_state : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    {
    "value": float,  # The mean SHAPr score across the sampled test instances, indicating the average similarity of test predictions to training predictions, where higher values suggest a greater risk of membership inference attacks.
    "sample_size": int # The number of samples used from the training and test sets to compute the SHAPr score, which is limited to a maximum of 5000 for efficiency.
    }
    '''
    if getattr(model, "_is_regressor", False) or (hasattr(y_train, "dtype") and y_train.dtype.kind == 'f'):
        raise ValueError("Not suitable for regression models or continuous targets. SHAPr is designed for classification tasks with discrete labels. It uses a kNN approach that relies on class labels to evaluate the similarity of test samples to training samples in the prediction space.")
    
    np.random.seed(random_state)

    sample_size = min(5000, len(X_train), len(X_test))

    idx_train = np.random.choice(len(X_train), sample_size, replace=False)
    idx_test  = np.random.choice(len(X_test), sample_size, replace=False)

    y_pred_train = model.predict(X_train.iloc[idx_train])
    y_pred_test  = model.predict(X_test.iloc[idx_test])

    train_size = 0.1

    phi = shapr_score(
        y_train[idx_train],
        y_test[idx_test],
        y_pred_train,
        y_pred_test,
        batch_size=200, # For efficiency. Matrix of 200 rows, one for each test sample with its k nearest neighbors in the training set.
        train_size=train_size  # For efficiency. 500-NN
    )

    mean_phi = float(np.mean(phi))

    if np.isnan(mean_phi) or np.isinf(mean_phi):
        raise ValueError("SHAPr calculation resulted in NaN or Inf. Check if your model and data are suitable for this metric.")
    return {
        "value": mean_phi,
        "sample_size": sample_size,
        "k_neighbors": sample_size * train_size,
    }


# =============================================================================
# Attribute Inference
# =============================================================================

def attribute_inference(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    sensitive_attribute: str,
) -> Dict[str, float]:
    '''
    Compute attribute inference risk for a specified sensitive attribute.
    The function evaluates how well an attacker can predict the sensitive attribute
    from the remaining features and the labels. It selects a linear regressor for continuous attributes and a 
    logistic classifier for categorical attributes, along with an appropriate metric
    (mean squared error for continuous, accuracy or F1 for categorical). 

    The function creates a BlackBoxAttack object that removes the target attribute
    from the training set, adds the label as a new input feature, and trains the attacker model.
    The trained attacker is then used to predict the removed attribute on the test set,
    and the predictions are compared with the true values using the selected metric function.

    The function returns a numeric value representing
    how predictable the attribute was based on the other features and labels, with
    higher values corresponding to greater risk of information leakage.
    
    Parameters
    ----------
    x_train : pd.DataFrame
        The training features.
    x_test : pd.DataFrame
        The testing features.
    y_train : pd.Series
        The training labels.
    y_test : pd.Series
        The testing labels.
    sensitive_attribute : str
        The name of the sensitive attribute column to evaluate for inference risk.

    Returns
    -------
    {
    float: A numeric value quantifying the attribute inference risk. Higher values
        indicate that the attribute is more easily predictable, representing
        a greater risk of sensitive information leakage.
    sensitive: The name of the sensitive attribute evaluated for inference risk (str).
    }
    '''
    if sensitive_attribute not in X_train.columns:
        raise ValueError(f"Sensitive attribute '{sensitive_attribute}' not found in training data columns.")
    
    # The library function 'to_numerical_or_categorical' requires pandas objects to use .astype("category")
    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)

    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)

    res = attribute_attack_score(
        X_train,
        X_test,
        y_train,
        y_test,
        sensitive_attribute,
    )

    is_continuous = X_train[sensitive_attribute].dtype.kind in ["i", "u", "f"]

    # If attribute is continous we invert the MSE score to have a consistent interpretation where higher values indicate higher risk
    if is_continuous:
        res = 1 / (1 + res)
    
    if np.isnan(res) or np.isinf(res):
        raise ValueError("Attribute inference calculation resulted in NaN or Inf. Check if your model and data are suitable for this metric.")

    return {
        "value": float(res),
        "sensitive": sensitive_attribute,
    }


# =============================================================================
# Privacy Risk
# =============================================================================

def privacy_risk(
    y_prob_train: np.ndarray,
    y_train: np.ndarray,
    y_prob_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    """
    Compute membership inference privacy risk for a machine learning model.

    This function estimates the likelihood that a specific data sample from the test set could be identified as 
    part of the training set of the target model, based directly on the model's predicted probabilities. 
    It computes a privacy risk score for each sample by comparing the behavior of training versus test samples 
    within the available predictions. The resulting score reflects the model's tendency to "memorize" its training data, 
    with higher values indicating that a sample exhibits characteristics similar to the training set, and therefore would be 
    more easily distinguished from test samples.

    A higher privacy risk score indicates that a sample is more distinguishable from 
    test data and, therefore, more vulnerable to membership inference attacks.

    Parameters
    ----------
    y_prob_train : np.ndarray
        Predicted probabilities of the target model on training samples.
    y_train : np.ndarray
        True labels for the training samples.
    y_prob_test : np.ndarray
        Predicted probabilities of the target model on test samples.
    y_test : np.ndarray
        True labels for the test samples.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the computed privacy risk under the key "value".
        Is the mean of the likelihood that an adversary could identify training samples based on the model's predicted probabilities, where
        higher values indicate a greater risk that an adversary could identify
        training samples, representing increased vulnerability to membership inference.

    References
    ----------
    .. [1] Song, L., & Mittal, P. (2021). Systematic evaluation of privacy risks of machine learning models. 
       In 30th USENIX Security Symposium (USENIX Security 21) (pp. 2615-2632).
    """
    if y_prob_train is None or y_prob_test is None:
        raise ValueError("Predicted probabilities for both training and test sets are required to compute privacy risk.")
    
    shadow_train = (y_prob_train, y_train)
    shadow_test  = (y_prob_test, y_test)
    target_train = (y_prob_train, y_train)

    # This function estimates the likelihood that an attacker could determine whether
    # a specific data sample was part of the training set of the target model, based 
    # on the model's predicted probabilities. It leverages distributions of a shadow 
    # model (a model trained on the same data) to compare the behavior of training versus test samples and computes a 
    # risk score for each training sample. The privacy risk score reflects the posterior
    # probability that a sample belongs to the training set given the model's output.
    scores = privacy_risk_score(
        shadow_train,
        shadow_test,
        target_train,
    )

    mean_score = float(np.mean(scores))

    if np.isnan(mean_score) or np.isinf(mean_score):
        raise ValueError("Privacy risk calculation resulted in NaN or Inf. Check if your model and data are suitable for this metric.")
    return {"value": mean_score}

# =============================================================================
# Accuracy Ratio (Data Minimization)
# =============================================================================

def accuracy_ratio(
    y_test: np.ndarray,
    y_pred_test: np.ndarray,
    model,
    X_test,
) -> Dict[str, float]:
    '''
    Compute accuracy ratio for data minimization techniques.

    This function evaluates the effectiveness of data minimization techniques by comparing 
    the model's performance on the original test set with its performance on a modified version 
    of the test set, where Gaussian noise has been added to the features. The accuracy ratio 
    is calculated as the ratio of the model's accuracy on the original test set to its accuracy 
    on the noisy test set. A ratio close to 1 indicates that the data minimization technique 
    has minimal impact on model performance, while a ratio greater than 1 suggests a loss in accuracy 
    due to the noise, reflecting a trade-off between privacy and utility.
    
    Parameters
    ----------
    y_test : np.ndarray
        The true labels for the test set.
    y_pred_test : np.ndarray
        The predicted labels for the original test set.
    model : object
        The trained model to evaluate.
    X_test : pd.DataFrame
        The original test features, which will be modified with noise for the data minimization evaluation.
    
    Returns
    -------
    Dict[str, float]
        A dictionary containing the computed accuracy ratio under the key "value", which quantifies the impact 
        of data minimization techniques on model performance, where values close to 1 indicate minimal performance loss.
    '''
    X_noisy = X_test + np.random.normal(0, 0.01, X_test.shape)
    y_pred_noisy = model.predict(X_noisy)

    y_pred_dm = [
        {
            "selector_type": "Noisy",
            "modifier_type": "GaussianNoise",
            "n_feats": X_test.shape[1],
            "feats": list(X_test.columns),
            "predictions": y_pred_noisy,
        }
    ]

    ratio = data_minimization_score(
        y_test,
        y_pred_test,
        y_pred_dm,
    )

    return {"value": float(ratio), "args": y_pred_dm[0]}


# =============================================================================
# k-Anonymity
# =============================================================================

def k_anonymity(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
) -> Dict[str, float]:
    """
    Compute k-anonymity for a dataset.

    This function evaluates the k-anonymity of a dataset by analyzing the combinations of specified quasi-identifiers. 
    It counts how many times each unique combination of quasi-identifiers appears in the dataset, and identifies the minimum 
    count across all combinations. The resulting k-anonymity value indicates the level of anonymity provided by the dataset, 
    where a higher k value means that each combination of quasi-identifiers is shared by more records, thus offering 
    stronger privacy protection against re-identification attacks.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset for which to compute k-anonymity.
    quasi_identifiers : List[str]
        A list of column names in the dataset that are considered quasi-identifiers.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the computed k-anonymity value under the key "value", along with the list of quasi-identifiers 
        used for the calculation. The k-anonymity value represents the minimum number of records that share the same combination 
        of quasi-identifiers, where higher values indicate stronger privacy protection.
    """
    counts = hol_k_anonymity(df, quasi_identifiers)
    # counts = df[quasi_identifiers].value_counts() # How many times each combination of quasi-identifiers appears.

    if isinstance(counts, pd.Series):
        k_value = counts.min() if not counts.empty else 0
    else:
        k_value = counts

    return {
        "value": float(k_value),
        "quasi_identifiers": quasi_identifiers,
    }


# =============================================================================
# l-Diversity
# =============================================================================

def l_diversity(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: List[str],
) -> Dict[str, float]:
    """
    Compute l-diversity for sensitive attributes in a dataset.

    It calculates the number of distinct values of each sensitive attribute for each 
    unique combination of quasi-identifiers, and identifies the minimum count across all groups. 
    The resulting l-diversity value indicates the level of diversity of sensitive attributes 
    within the dataset, where a higher l value means that there are more distinct values of the 
    sensitive attributes within each group, thus offering stronger privacy protection against attribute disclosure attacks.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset for which to compute l-diversity.
    quasi_identifiers : List[str]
        A list of column names in the dataset that are considered quasi-identifiers.
    sensitive_attributes : List[str]
        A list of column names in the dataset that are considered sensitive attributes.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the computed l-diversity value under the key "value", along with the lists of quasi-identifiers and sensitive attributes used for the calculation. The l-diversity value represents the minimum number of distinct values of the sensitive attributes within groups defined by the quasi-identifiers, where higher values indicate stronger privacy protection against attribute disclosure.
    """
    result = hol_l_diversity(df, quasi_identifiers, sensitive_attributes)
    # df_grouped = df.groupby(quasi_identifiers, as_index=False)
    # result = {
    #         s: sorted([len(row["unique"]) for _, row in df_grouped[s].agg(["unique"]).dropna().iterrows()])
    #         for s in sensitive_attributes
    #     }
    
    all_vals = []

    if isinstance(result, dict):
        for v in result.values():
            if isinstance(v, list):
                all_vals.extend(v)

    min_l = min(all_vals) if all_vals else 0

    return {
        "value": float(min_l),
        "quasi_identifiers": quasi_identifiers,
        "sensitive_attributes": sensitive_attributes,
    }


# =============================================================================
# t-Closeness
# =============================================================================

def t_closeness(
    df: pd.DataFrame,
    quasi_identifiers: List[str],
    sensitive_attributes: List[str],
) -> Dict[str, float]:
    """
    Compute t-closeness for sensitive attributes using Earth Mover's Distance (EMD) between local and global distributions.

    It calculates the Earth Mover's Distance (EMD) between the distribution of each sensitive attribute within groups defined 
    by the quasi-identifiers and the overall distribution of that attribute in the entire dataset. The function iterates through 
    each sensitive attribute and each group of records sharing the same combination of quasi-identifiers, computes the local 
    distribution of the sensitive attribute within that group, and compares it to the global distribution using EMD. 
    The maximum EMD value across all groups and sensitive attributes is returned as the t-closeness score, where lower 
    values indicate that the distribution of sensitive attributes within groups is closer to the overall distribution, 
    thus providing stronger privacy protection against attribute disclosure attacks.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset for which to compute t-closeness.
    quasi_identifiers : List[str]
        A list of column names in the dataset that are considered quasi-identifiers.
    sensitive_attributes : List[str]
        A list of column names in the dataset that are considered sensitive attributes.

    Returns
    -------
    Dict[str, float]
        A dictionary containing the computed t-closeness value under the key "value", along with the lists of quasi-identifiers 
        and sensitive attributes used for the calculation. The t-closeness value represents the maximum Earth Mover's Distance 
        between the local distribution of sensitive attributes within groups defined by the quasi-identifiers and the global 
        distribution of those attributes in the entire dataset, where lower values indicate stronger privacy protection against attribute disclosure.
    """
    max_t = 0.0

    for s in sensitive_attributes:

        global_dist = df[s].value_counts(normalize=True)

        grouped = df.groupby(quasi_identifiers)

        for _, group in grouped:

            local_dist = group[s].value_counts(normalize=True)

            aligned = global_dist.index.union(local_dist.index)
            
            # tvd = 0.5 * np.sum(np.abs(g - l)) # Total Variation Distance.
            # distances.append(tvd)
            g = global_dist.reindex(aligned, fill_value=0)
            l = local_dist.reindex(aligned, fill_value=0)

            g_cdf = np.cumsum(g.values)
            l_cdf = np.cumsum(l.values)

            emd = float(np.sum(np.abs(g_cdf - l_cdf)))
            max_t = max(max_t, emd)

    return {   
        "value": max_t, 
        "quasi_identifiers": quasi_identifiers, 
        "sensitive_attributes": sensitive_attributes}

