```mermaid
classDiagram
  direction LR

  class TrustEvaluator {
    +model
    +train_data
    +test_data
    +factsheet
    +output_path: str
    +config: dict
    +context: EvaluationContext
    +result: dict | None
    +evaluate(pillars: list[str], show_nan: bool) dict
    +plot_results() None
    +plot_radar() None
    +compare_all_bars(results: dict[str, dict]) None
    +compare_radar(results: dict[str, dict]) None
    -_load_config(config_path: str | None) dict
    -_build_context() EvaluationContext
    -_predict(X)
    -_predict_proba(X)
    -_compute_pillar_scores(pillars: list[str] | None) dict
    -_compute_trust_score(pillar_scores: dict) float
    -_build_score_explanation(pillar_results: dict, show_nan: bool) dict
    -_save_result() None
    -_is_nan(value)
  }

  class BaseMetric {
    <<abstract>>
    +metric_key: str
    +score_config_key: str | None
    +evaluate(context, config: dict | None) Result
    +compute(context) dict
    +build_properties(raw: dict) dict
    +compute_score(raw: dict, config: dict | None) int
    +custom_score(raw: dict)
    -_get_thresholds(config: dict | None)
  }

  class Pillar {
    <<abstract>>
    +pillar_key: str
    +prepare(context: EvaluationContext, config: dict[str, Any]) None
    +get_metrics() List~BaseMetric~
    +analyse(context: EvaluationContext, config: dict[str, dict]) Result
    +score(context: EvaluationContext, config: dict) Tuple~float, Result~
  }

  class EvaluationContext {
    <<dataclass>>
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
    +group_mask: ndarray
    +protected_feature: str
    +protected_values: list[Any]
    +target_column: str
    +favorable_outcomes: list[Any]
    +y_prob_positive: ndarray | None
  }

  class Result {
    <<NamedTuple>>
    +score
    +properties
  }

  class FairnessPillar {
    +pillar_key: str
    +prepare(context: EvaluationContext, config: dict[str, Any]) None
    +get_metrics() List~BaseMetric~
  }

  class AccountabilityPillar {
    +pillar_key: str
    +prepare(context: EvaluationContext, config: dict[str, Any]) None
    +get_metrics() List~BaseMetric~
  }

  class PrivacyPillar {
    +pillar_key: str
    +prepare(context: EvaluationContext, config: dict[str, Any]) None
    +get_metrics() List~BaseMetric~
  }

  class SustainabilityPillar {
    +pillar_key: str
    +prepare(context: EvaluationContext, config: dict[str, Any]) None
    +get_metrics() List~BaseMetric~
  }

  class ExplainabilityPillar {
    +pillar_key: str
    +prepare(context: EvaluationContext, config: dict[str, Any]) None
    +get_metrics() List~BaseMetric~
  }

  class RobustnessPillar {
    +pillar_key: str
    +prepare(context: EvaluationContext, config: dict[str, Any]) None
    +get_metrics() List~BaseMetric~
  }

  TrustEvaluator *-- EvaluationContext : builds
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


```