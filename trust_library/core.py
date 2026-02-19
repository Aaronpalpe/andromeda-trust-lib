import json
import os

from trust_library import accountability, privacy, sustainability, utils
from trust_library.fairness import fairness

class TrustEvaluator:
    def __init__(self, model, train_data, test_data, factsheet, config_path="trust_library/configs.json"):
        """
        Inicializa el evaluador.
        Args:
            model: Modelo cargado (sklearn/keras/pkl).
            train_data: DataFrame de entrenamiento.
            test_data: DataFrame de test.
            factsheet: Diccionario con metadatos (protected_feature, target, etc).
            config_path: Ruta al json de configuración.
        """
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.factsheet = factsheet
        
        # Cargar Configuración
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError(f"Configuración no encontrada en: {config_path}")
            
        self.results_per_pillar = {}
        self.pillar_score = {}
        self.final_trust_score = 0

    def compute(self):
        """Ejecuta los cálculos de los pilares."""
        mappings = self.config.get("mappings", {})
        
        pillars = {
            "fairness": fairness,
            "privacy": privacy,
            "accountability": accountability,
            "sustainability": sustainability,
        }

        target = self.factsheet["general"]["target_column"]["value"]
        if target not in self.train_data.columns or target not in self.test_data.columns:
            raise ValueError(f"Target column '{target}' no encontrada en los datasets.")
        
        X_train = self.train_data.drop(columns=[target])
        y_train = self.train_data[target].values.flatten()
        X_test = self.test_data.drop(columns=[target])
        y_test = self.test_data[target].values.flatten()
        
        # Hacemos las predicciones para el train y test 
        try:
            y_pred_train = self.model.predict(X_train)
            y_pred_test = self.model.predict(X_test)

            if hasattr(y_pred_train, 'flatten'): 
                y_pred_train = y_pred_train.flatten()
            if hasattr(y_pred_test, 'flatten'):
                y_pred_test = y_pred_test.flatten()
            # Calcular probabilidades si el modelo lo soporta 
            y_prob_train = None
            if hasattr(self.model, "predict_proba"):
                y_prob_train = self.model.predict_proba(X_train)
            y_prob_test = None
            if hasattr(self.model, "predict_proba"):
                y_prob_test = self.model.predict_proba(X_test)

        except Exception as e:
            raise RuntimeError(f"Error en la predicción del modelo: {e}")

        context = utils.EvaluationContext(
            model=self.model,
            train_data=self.train_data,
            test_data=self.test_data,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            y_prob_train=y_prob_train,
            y_prob_test=y_prob_test,
            factsheet=self.factsheet
        )

        for pillar_name, pillar_module in pillars.items():
            print(f"Calculando métricas de {pillar_name.capitalize()}...")
            
            self.results_per_pillar[pillar_name] = pillar_module.analyse(
                context,
                mappings.get(pillar_name), 
            )

        # 2. Calcular Scores Ponderados
        metrics_and_scores_per_pillar = {k: v.score for k, v in self.results_per_pillar.items()}
        metrics_properties_per_pillar = {k: v.properties for k, v in self.results_per_pillar.items()}

        weights_config_per_pillar = self.config.get("weights", {})
        
        for pillar, metrics_and_scores_in_pillar in self.results_per_pillar.items():
            weights_in_pillar = weights_config_per_pillar.get(pillar, {})
            self.pillar_score[pillar] = utils.calculate_weighted_score(metrics_and_scores_in_pillar.score, weights_in_pillar)
            
        # 3. Calcular Trust Score Global
        pillar_weights = self.config.get("pillars", {})
        self.final_trust_score = utils.calculate_weighted_score(self.pillar_score, pillar_weights)
        
        res = {
            "trust_score": self.final_trust_score,
            "pillar_score": self.pillar_score,
            "details": metrics_and_scores_per_pillar,
            "properties": metrics_properties_per_pillar 
        }

        #convertimos a json
        json_res = utils.to_json_safe(res)

        # Guardamos el resultado en un json
        with open("trust_evaluation_result.json", 'w') as f:
            json.dump(json_res, f, indent=4)
        return res