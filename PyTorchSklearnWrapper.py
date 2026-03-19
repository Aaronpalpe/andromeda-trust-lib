import torch
import torch.nn as nn
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class PyTorchSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper para usar modelos de PyTorch como si fueran estimadores de scikit-learn.
    Ideal para integrarse en pipelines de evaluación (TrustEvaluator, SHAP, etc.).
    """
    
    def __init__(self, model: nn.Module, device: str = "cpu", num_classes: int = 2):
        self.model = model
        self.device = torch.device(device)
        self.num_classes = num_classes
        self.model.to(self.device)
        self.is_fitted_ = True 
        self.classes_ = np.array([0, 1])
        
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

    def predict_proba(self, X):
        """
        Calcula las probabilidades usando Softmax (o Sigmoid para binaria).
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        
        # Modo inferencia
        self.model.eval()
        
        # Convertir datos a tensores de PyTorch
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            
            # Si es clasificación multiclase, usamos Softmax
            # Si el modelo ya escupe probabilidades, podrías omitir esto.
            probabilities = torch.softmax(outputs, dim=1) 
            
        return probabilities.cpu().numpy()

    def predict(self, X):
        """
        Calcula la clase final basándose en la probabilidad más alta.
        """
        probas = self.predict_proba(X)
        # Obtenemos el índice de la probabilidad más alta
        predictions = np.argmax(probas, axis=1)
        
        # Mapeamos al nombre original de las clases si es necesario
        if hasattr(self, 'classes_'):
            return self.classes_[predictions]
        return predictions