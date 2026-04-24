@startuml
left to right direction

class TrustEvaluator {
  +model: Any
  +train_data: DataFrame
  +test_data: DataFrame
  +factsheet: dict | Factsheet
  +output_path: str
  +config: dict
  +context: EvaluationContext
  +result: dict | None
  +set_global_seed(seed: int): None
  +evaluate(pillars: list[str] = None, show_nan: bool = False): dict
  +plot_results(): None
  +plot_radar(): None
  +print_results_summary(decimals: int = 3): None
  +compare_all_bars(results: dict[str, dict]): None
  +compare_radar(results: dict[str, dict]): None
  -_load_config(config_path: str | None): dict
  -_build_context(): EvaluationContext
  -_predict(X): Any
  -_predict_proba(X): Any | None
  -_compute_pillar_scores(pillars: list[str] | None = None): dict
  -_compute_trust_score(pillar_scores: dict): float
  -_build_score_explanation(pillar_results: dict, show_nan: bool = False): dict
  -_save_result(): None
  -_save_updated_factsheet(): None
  -_validate_pillars(pillars: list[str]): None
  -_is_nan(value): bool
}

class Factsheet {
  +__init__(general: dict = None, governance: dict = None, auditability: dict = None, redressability: dict = None, fairness: dict = None, privacy: dict = None, sustainability: dict = None, load_path: str = None, save_path: str = None)
  +to_dict(): dict
  +save(path: str = "factsheet.json"): None
  +create_factsheet_interactive(output_path: str = "factsheet.json"): Factsheet
  +__getitem__(key): Any
  +get(key, default = None): Any
  -_load_template(): dict
  -_apply_section(section_name: str, values: dict): None
}

abstract class BaseMetric {
  +metric_key: str
  +score_config_key: str | None
  +evaluate(context, config: dict | None): Result
  +compute(context): dict
  +build_properties(raw: dict): dict
  +compute_score(raw: dict, config: dict | None): float
  +custom_score(raw: dict): Any
  -_get_thresholds(config: dict | None): list[float] | None
  -_get_normalized_config(config: dict | None): dict | None
}

abstract class Pillar {
  +pillar_key: str
  +prepare(context: EvaluationContext, config: dict[str, Any]): None
  +get_metrics(): List~BaseMetric~
  +analyse(context: EvaluationContext, config: dict[str, dict]): Result
  +score(context: EvaluationContext, config: dict): Tuple~float, Result~
}

class EvaluationContext <<dataclass>> {
  +model: Any
  +train_data: DataFrame
  +test_data: DataFrame
  +X_train: DataFrame
  +X_test: DataFrame
  +y_train: ndarray
  +y_test: ndarray
  +y_pred_train: ndarray | DataFrame
  +y_pred_test: ndarray | DataFrame
  +y_prob_train: ndarray | DataFrame | None
  +y_prob_test: ndarray | DataFrame | None
  +factsheet: dict[str, Any]
  +extras: dict[str, Any]
  +group_mask: ndarray <<property>>
  +protected_feature: str <<property>>
  +protected_values: list[Any] <<property>>
  +target_column: str <<property>>
  +favorable_outcomes: list[Any] <<property>>
  +y_prob_positive: ndarray | None <<property>>
}

class Result <<namedtuple>> {
  +score: Any
  +properties: Any
}

class FairnessPillar {
  +pillar_key: str
  +prepare(context: EvaluationContext, config: dict[str, Any]): None
  +get_metrics(): List~BaseMetric~
}

class AccountabilityPillar {
  +pillar_key: str
  +prepare(context: EvaluationContext, config: dict[str, Any]): None
  +get_metrics(): List~BaseMetric~
}

class PrivacyPillar {
  +pillar_key: str
  +prepare(context: EvaluationContext, config: dict[str, Any]): None
  +get_metrics(): List~BaseMetric~
}

class SustainabilityPillar {
  +pillar_key: str
  +prepare(context: EvaluationContext, config: dict[str, Any]): None
  +get_metrics(): List~BaseMetric~
}

class ExplainabilityPillar {
  +pillar_key: str
  +prepare(context: EvaluationContext, config: dict[str, Any]): None
  +get_metrics(): List~BaseMetric~
}

class RobustnessPillar {
  +pillar_key: str
  +prepare(context: EvaluationContext, config: dict[str, Any]): None
  +get_metrics(): List~BaseMetric~
}

class FairnessMetrics <<metrics>>
class AccountabilityMetrics <<metrics>>
class PrivacyMetrics <<metrics>>
class SustainabilityMetrics <<metrics>>
class ExplainabilityMetrics <<metrics>>
class RobustnessMetrics <<metrics>>

TrustEvaluator *-- EvaluationContext : builds
TrustEvaluator ..> Factsheet : accepts/updates
TrustEvaluator ..> FairnessPillar : _PILLARS
TrustEvaluator ..> AccountabilityPillar : _PILLARS
TrustEvaluator ..> PrivacyPillar : _PILLARS
TrustEvaluator ..> SustainabilityPillar : _PILLARS
TrustEvaluator ..> ExplainabilityPillar : _PILLARS
TrustEvaluator ..> RobustnessPillar : _PILLARS

TrustEvaluator ..> Result : returns
Pillar ..> Result : analyse()/score()
BaseMetric ..> Result : evaluate()
Pillar ..> EvaluationContext : prepare/analyse/score
BaseMetric ..> EvaluationContext : compute(context)

FairnessPillar --|> Pillar
AccountabilityPillar --|> Pillar
PrivacyPillar --|> Pillar
SustainabilityPillar --|> Pillar
ExplainabilityPillar --|> Pillar
RobustnessPillar --|> Pillar

FairnessPillar ..> FairnessMetrics : get_metrics()
AccountabilityPillar ..> AccountabilityMetrics : get_metrics()
PrivacyPillar ..> PrivacyMetrics : get_metrics()
SustainabilityPillar ..> SustainabilityMetrics : get_metrics()
ExplainabilityPillar ..> ExplainabilityMetrics : get_metrics()
RobustnessPillar ..> RobustnessMetrics : get_metrics()

@enduml