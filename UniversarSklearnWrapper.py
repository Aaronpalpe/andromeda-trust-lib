import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted

class UniversalSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper universal para usar modelos de PyTorch, TensorFlow, scikit-learn 
    y otros como si fueran estimadores estándar de scikit-learn.
    Ideal para integrarse en pipelines de evaluación (TrustEvaluator, SHAP, etc.).
    """
    
    def __init__(self, model, device: str = "cpu", num_classes: int = 2):
        self.model = model
        self.num_classes = num_classes
        self.device_name = device
        self.framework = self._detect_framework(model)

        # Configuración específica si es PyTorch
        if self.framework == 'pytorch':
            import torch
            self.device = torch.device(self.device_name)
            self.model.to(self.device)

        # Atributos requeridos por scikit-learn para considerar el modelo como entrenado
        self.is_fitted_ = True
        self.classes_ = np.arange(self.num_classes)
        self.n_features_in_ = None  # Set dynamically on first predict call
        self._estimator_type = "classifier"  # Required for sklearn's partial_dependence

    def _detect_framework(self, model):
        """
        Identifica el framework leyendo el módulo directo o inspeccionando 
        las clases base (MRO) para soportar modelos definidos en __main__.
        """
        module_name = type(model).__module__.lower()
        
        # Extraemos los módulos de todas las clases de las que hereda este modelo
        mro_modules = [base.__module__.lower() for base in type(model).__mro__]
        
        # 1. Búsqueda profunda en la herencia (Resuelve el problema de __main__)
        if any(mod.startswith('torch') for mod in mro_modules):
            return 'pytorch'
        elif any(mod.startswith('sklearn') for mod in mro_modules):
            return 'sklearn'
        elif any(mod.startswith('tensorflow') or mod.startswith('keras') for mod in mro_modules):
            return 'tensorflow'
        elif any(mod.startswith('cntk') for mod in mro_modules):
            return 'cntk'
            
        # 2. Fallback al chequeo original por si es un objeto instanciado raro
        if module_name.startswith('sklearn'):
            return 'sklearn'
        elif module_name.startswith('tensorflow') or module_name.startswith('keras'):
            return 'tensorflow'
        elif module_name.startswith('torch'):
            return 'pytorch'
        elif hasattr(model, 'predict'):
            # Fallback para XGBoost, LightGBM, CatBoost o modelos custom
            return 'generic_predict'
        else:
            raise ValueError(f"No se pudo identificar un framework compatible. Módulos detectados: {mro_modules}")

    # def fit(self, X, y, epochs=10, lr=0.001):
    #     """
    #     Implementación básica de fit. 
    #     Si el modelo ya viene entrenado (como en tu caso), este método puede 
    #     ser ignorado o simplemente retornar self, pero lo incluimos por cumplir el contrato.
    #     """
    #     X, y = check_X_y(X, y)
    #     self.classes_ = np.unique(y) # Necesario para scikit-learn
        
    #     # Opcional: Lógica real de entrenamiento si lo necesitas en el futuro.
    #     # Por ahora, asumimos que el modelo ya está entrenado (loaded_model).
    #     self.is_fitted_ = True 
    #     return self

    def fit(self, X, y=None, **kwargs):
        """
        Método dummy para cumplir con el contrato de BaseEstimator de scikit-learn.
        Como el modelo ya está preentrenado, simplemente validamos entradas básicas
        y retornamos self.
        """
        # Opcional: Validar que los datos de entrada tienen sentido
        X = check_array(X)
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]

        if y is not None:
            self.classes_ = np.unique(y)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Calcula las probabilidades adaptándose al framework subyacente.
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
                raise AttributeError(f"El modelo de {self.framework} no implementa 'predict_proba'.")

        elif self.framework == 'tensorflow':
            # En TF/Keras predict() para clasificación ya suele devolver probabilidades
            probas = self.model.predict(X, verbose=0)
            # Si es clasificación binaria y devuelve un array de shape (N, 1),
            # lo convertimos al estándar de sklearn (N, 2)
            if probas.shape[1] == 1 and self.num_classes == 2:
                probas = np.hstack([1 - probas, probas])
            return probas

        elif self.framework == 'pytorch':
            import torch
            self.model.eval()

            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                outputs = self.model(X_tensor)
                # Usamos Softmax para multiclase/binaria (asumiendo que el modelo devuelve logits)
                probabilities = torch.softmax(outputs, dim=1)

            return probabilities.cpu().numpy()

        elif self.framework == 'cntk':
            input_var = self.model.arguments[0]
            outputs = self.model.eval({input_var: X})
            # Implementación softmax manual simple usando numpy si cntk devuelve logits
            e_x = np.exp(outputs - np.max(outputs, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)

    def predict(self, X):
        """
        Calcula la clase final. Para Deep Learning usa argmax sobre predict_proba,
        para ML tradicional usa el predict nativo.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)

        # Set n_features_in_ on first call for sklearn compatibility
        if self.n_features_in_ is None:
            self.n_features_in_ = X.shape[1]

        # Para sklearn y modelos genéricos (XGBoost, etc.), usamos su predict nativo
        if self.framework in ['sklearn', 'generic_predict']:
            return self.model.predict(X)

        # Para Deep Learning (PyTorch, TF, CNTK), derivamos la clase de las probabilidades
        probas = self.predict_proba(X)
        predictions = np.argmax(probas, axis=1)

        # Mapeamos al nombre original de las clases
        if hasattr(self, 'classes_'):
            return self.classes_[predictions]

        return predictions