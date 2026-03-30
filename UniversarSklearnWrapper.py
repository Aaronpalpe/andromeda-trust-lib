import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted

class UniversalSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    Universal wrapper to use models from PyTorch, TensorFlow, scikit-learn
    and others as if they were standard scikit-learn estimators.
    Ideal for integration into evaluation pipelines (TrustEvaluator, SHAP, etc.).
    """
    
    def __init__(self, model, device: str = "cpu", num_classes: int = 2):
        self.model = model
        self.num_classes = num_classes
        self.device_name = device
        self.framework = self._detect_framework(model)

        # Framework-specific configuration for PyTorch models
        if self.framework == 'pytorch':
            import torch
            self.device = torch.device(self.device_name)
            self.model.to(self.device)

        # Attributes required by scikit-learn to consider the model as trained
        self.is_fitted_ = True
        self.classes_ = np.arange(self.num_classes)
        self.n_features_in_ = None  # Set dynamically on first predict call
        self._estimator_type = "classifier"  # Required for sklearn's partial_dependence

    def _detect_framework(self, model):
        """
        Identifies the framework by reading the module directly or inspecting
        base classes (MRO) to support models defined in __main__.
        """
        module_name = type(model).__module__.lower()

        # Extract modules from all classes inherited by this model
        mro_modules = [base.__module__.lower() for base in type(model).__mro__]

        # 1. Deep search in inheritance (Resolves the __main__ problem)
        if any(mod.startswith('torch') for mod in mro_modules):
            return 'pytorch'
        elif any(mod.startswith('sklearn') for mod in mro_modules):
            return 'sklearn'
        elif any(mod.startswith('tensorflow') or mod.startswith('keras') for mod in mro_modules):
            return 'tensorflow'
        elif any(mod.startswith('cntk') for mod in mro_modules):
            return 'cntk'

        # 2. Fallback to original check in case of unusual instantiated objects
        if module_name.startswith('sklearn'):
            return 'sklearn'
        elif module_name.startswith('tensorflow') or module_name.startswith('keras'):
            return 'tensorflow'
        elif module_name.startswith('torch'):
            return 'pytorch'
        elif hasattr(model, 'predict'):
            # Fallback for XGBoost, LightGBM, CatBoost or custom models
            return 'generic_predict'
        else:
            raise ValueError(f"Could not identify a compatible framework. Detected modules: {mro_modules}")

    # def fit(self, X, y, epochs=10, lr=0.001):
    #     """
    #     Basic fit implementation.
    #     If the model is already trained (as in your case), this method can
    #     be ignored or simply return self, but we include it to comply with the contract.
    #     """
    #     X, y = check_X_y(X, y)
    #     self.classes_ = np.unique(y) # Required for scikit-learn

    #     # Optional: Real training logic if you need it in the future.
    #     # For now, we assume the model is already trained (loaded_model).
    #     self.is_fitted_ = True
    #     return self

    def fit(self, X, y=None, **kwargs):
        """
        Dummy method to comply with scikit-learn's BaseEstimator contract.
        Since the model is already pre-trained, we simply validate basic inputs
        and return self.
        """
        # Optional: Validate that input data makes sense
        X = check_array(X)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]

        if y is not None:
            self.classes_ = np.unique(y)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Computes probabilities adapting to the underlying framework.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        # Set n_features_in_ on first call for sklearn compatibility
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]

        if self.framework in ['sklearn', 'generic_predict']:
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                raise AttributeError(f"The {self.framework} model does not implement 'predict_proba'.")

        elif self.framework == 'tensorflow':
            # In TF/Keras predict() for classification typically returns probabilities
            probas = self.model.predict(X, verbose=0)
            # If it's binary classification and returns an array of shape (N, 1),
            # we convert it to sklearn's standard (N, 2)
            if probas.shape[1] == 1 and self.num_classes == 2:
                probas = np.hstack([1 - probas, probas])
            return probas

        elif self.framework == 'pytorch':
            import torch
            self.model.eval()

            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                outputs = self.model(X_tensor)
                # Use Softmax for multiclass/binary (assuming the model returns logits)
                probabilities = torch.softmax(outputs, dim=1)

            return probabilities.cpu().numpy()

        elif self.framework == 'cntk':
            input_var = self.model.arguments[0]
            outputs = self.model.eval({input_var: X})
            # Simple manual softmax implementation using numpy if cntk returns logits
            e_x = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)

    def predict(self, X):
        """
        Computes the final class. For Deep Learning uses argmax on predict_proba,
        for traditional ML uses native predict.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        # Set n_features_in_ on first call for sklearn compatibility
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]

        # For sklearn and generic models (XGBoost, etc.), use their native predict
        if self.framework in ['sklearn', 'generic_predict']:
            return self.model.predict(X)

        # For Deep Learning (PyTorch, TF, CNTK), derive the class from probabilities
        probas = self.predict_proba(X)
        predictions = np.argmax(probas, axis=1)

        # Map to original class names
        if hasattr(self, 'classes_'):
            return self.classes_[predictions]

        return predictions