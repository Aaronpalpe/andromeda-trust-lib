# import copy
# import json
# from pathlib import Path

# from trust_library.utils import to_json_safe

# # Default path to the base configuration file
# DEFAULT_CONFIG_FILE_PATH = Path(__file__).parent / "configs.json"


# class Config:
#     """
#     Domain object representing the Trust configuration.

#     Supports three creation modes:
#     - From default template
#     - From existing file (path)
#     """

#     def __init__(self, load_path: str = None): # config: dict = None, 
#         """
#         Initialize the Config object.

#         Args:
#             load_path (str): Path to a JSON config file
#         """
#         self._template = self._load_default()

#         if load_path:
#             # Load configuration from file
#             with open(load_path, "r", encoding="utf-8") as f:
#                 self._config = json.load(f)
#         # elif config:
#         #     # Use provided dictionary
#         #     self._config = config
#         else:
#             # Create a fresh copy from the default template
#             self._config = copy.deepcopy(self._template)

#     def _load_default(self):
#         """
#         Load the default configuration template.

#         Returns:
#             dict: Default configuration structure
#         """
#         with open(DEFAULT_CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
#             return json.load(f)

#     def to_dict(self):
#         """
#         Return the internal configuration dictionary.

#         Returns:
#             dict: Configuration data
#         """
#         return self._config

#     def save(self, path="configs.json"):
#         """
#         Save the configuration to a JSON file.

#         Args:
#             path (str): Output file path
#         """
#         with open(path, "w", encoding="utf-8") as f:
#             json.dump(
#                 to_json_safe(self._config),
#                 f,
#                 indent=4,
#                 ensure_ascii=False
#             )


# # # -------------------------
# # # Utility helper
# # # -------------------------

# # def _get_config_dict(config):
# #     """
# #     Extract the internal dictionary from a Config object
# #     or return the dict directly if already a dict.

# #     Args:
# #         config (Config | dict)

# #     Returns:
# #         dict
# #     """
# #     return config._config if isinstance(config, Config) else config


# # # -------------------------
# # # Load / Save functions
# # # -------------------------

# # def load_config_default(config_path=DEFAULT_CONFIG_FILE_PATH):
# #     """
# #     Load the default configuration as a Config object.

# #     Args:
# #         config_path (str): Path to default config

# #     Returns:
# #         Config
# #     """
# #     return Config(load_path=config_path)


# # def load_config(config_path="configs.json"):
# #     """
# #     Load a configuration file as a Config object.

# #     Args:
# #         config_path (str): Path to config file

# #     Returns:
# #         Config
# #     """
# #     return Config(load_path=config_path)


# # def save_config(config, config_path="configs.json"):
# #     """
# #     Save a configuration (dict or Config) to file.

# #     Args:
# #         config (dict | Config): Configuration to save
# #         config_path (str): Output file path
# #     """
# #     config_dict = _get_config_dict(config)

# #     with open(config_path, "w", encoding="utf-8") as f:
# #         json.dump(
# #             to_json_safe(config_dict),
# #             f,
# #             indent=4,
# #             ensure_ascii=False
# #         )


# # # -------------------------
# # # Configuration modifiers
# # # -------------------------

# # def set_pillar_weight(config, pillar_name, new_weight):
# #     """
# #     Modify the weight of a pillar.

# #     Ensures that the total sum of pillar weights remains 1.

# #     Args:
# #         config (dict | Config)
# #         pillar_name (str)
# #         new_weight (float)

# #     Returns:
# #         dict: Updated configuration
# #     """
# #     config = _get_config_dict(config)

# #     if pillar_name not in config["pillars"]:
# #         raise ValueError(f"Pillar '{pillar_name}' does not exist.")

# #     # total = sum(config["pillars"].values()) - config["pillars"][pillar_name] + new_weight

# #     # if total != 1:
# #     #     raise ValueError("Total pillar weights must sum to 1. Change not applied.")
# #     # else:
# #     config["pillars"][pillar_name] = float(new_weight)

# #     return config


# # def set_metric_weight(config, pillar_name, metric_name, new_weight):
# #     """
# #     Modify the weight of a metric within a pillar.

# #     Ensures that the sum of metric weights in that pillar remains 1.

# #     Args:
# #         config (dict | Config)
# #         pillar_name (str)
# #         metric_name (str)
# #         new_weight (float)

# #     Returns:
# #         dict: Updated configuration
# #     """
# #     config = _get_config_dict(config)

# #     if pillar_name not in config["weights"]:
# #         raise ValueError(f"Pillar '{pillar_name}' does not exist.")

# #     if metric_name not in config["weights"][pillar_name]:
# #         raise ValueError(f"Metric '{metric_name}' does not exist in '{pillar_name}'.")

# #     # total = (
# #     #     sum(config["weights"][pillar_name].values())
# #     #     - config["weights"][pillar_name][metric_name]
# #     #     + new_weight
# #     # )

# #     # if total != 1:
# #     #     raise ValueError("Metric weights must sum to 1 within the pillar. Change not applied.")
# #     # else:
# #     config["weights"][pillar_name][metric_name] = float(new_weight)

# #     return config


# # def set_metric_thresholds(config, pillar_name, score_metric_name, new_thresholds):
# #     """
# #     Modify threshold values for a metric.

# #     Args:
# #         config (dict | Config)
# #         pillar_name (str)
# #         score_metric_name (str)
# #         new_thresholds (list): Must contain exactly 4 values

# #     Returns:
# #         dict: Updated configuration
# #     """
# #     config = _get_config_dict(config)

# #     if len(new_thresholds) != 4:
# #         raise ValueError("new_thresholds must have length 4.")

# #     try:
# #         config["mappings"][pillar_name][score_metric_name]["thresholds"]["value"] = list(new_thresholds)
# #     except KeyError:
# #         raise ValueError(
# #             f"Thresholds not found for {score_metric_name} in {pillar_name}"
# #         )

# #     return config


# # def set_metric_mapping_values(config, pillar_name, score_metric_name, new_mapping_dict):
# #     """
# #     Modify categorical mapping values for a metric.

# #     Args:
# #         config (dict | Config)
# #         pillar_name (str)
# #         score_metric_name (str)
# #         new_mapping_dict (dict): Must contain exactly 5 elements

# #     Returns:
# #         dict: Updated configuration
# #     """
# #     config = _get_config_dict(config)

# #     if len(new_mapping_dict) != 5:
# #         raise ValueError("new_mapping_dict must have length 5.")

# #     try:
# #         config["mappings"][pillar_name][score_metric_name]["mappings"]["value"] = dict(new_mapping_dict)
# #     except KeyError:
# #         raise ValueError(
# #             f"Mappings not found for {score_metric_name} in {pillar_name}"
# #         )

# #     return config


# # # -------------------------
# # # Interactive creation (optional)
# # # -------------------------

# # def create_config_interactive(output_path="configs.json"):
# #     """
# #     Create a configuration interactively via terminal input.

# #     Only modifies pillar weights (simple example).

# #     Args:
# #         output_path (str): File where the config will be saved

# #     Returns:
# #         Config
# #     """
# #     template = Config()._load_default()
# #     config = copy.deepcopy(template)

# #     print("\n=== INTERACTIVE CONFIGURATION ===")

# #     for pillar in config["pillars"]:
# #         current = config["pillars"][pillar]
# #         user_input = input(f"Weight for '{pillar}' (current={current}): ").strip()

# #         if user_input:
# #             config["pillars"][pillar] = float(user_input)

# #     with open(output_path, "w", encoding="utf-8") as f:
# #         json.dump(config, f, indent=4, ensure_ascii=False)

# #     print(f"\nConfiguration saved to {output_path}")

# #     return Config(path=output_path)