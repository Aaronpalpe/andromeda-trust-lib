import copy
import json
from pathlib import Path

from trust_library.utils import to_json_safe

DEFAULT_FACTSHEET_FILE_PATH = Path(__file__).parent / "factsheet.json"


class Factsheet:
    """
    Domain object representing a Trust Library Factsheet.

    The constructor accepts structured inputs per section and injects them
    into the official template, ensuring schema compliance.

    Example usage:
        ### 1. From scratch
        factsheet = Factsheet(
            general={
                "model_name": "Credit Risk Model",
                "purpose_description": "Predict loan default risk",
                "domain_description": "Finance",
                "training_data_description": "Banking dataset with demographic features",
                "model_information": "DecisionTreeClassifier depth=5",
                "authors": ["Alice", "Bob"],
                "contact_information": "ml-team@company.com"
            },
            fairness={
                "protected_feature": "Group",
                "protected_values": [1],
                "favorable_outcomes": [1]
            },
            privacy={
                "epsilon": 0.5,
                "quasi_identifiers": ["Own_Housing", "Own_Car"],
                "sensitive_attribute": ["Group"]
            }
        )

        ### 2. From file
        fs = Factsheet(load_path="factsheet.json")

        ### 3. Interactive
        fs = create_factsheet_interactive()

        ### Save to file
        fs.save(path="my_factsheet.json")
    """

    def __init__(
        self,
        general: dict = None,
        governance: dict = None,
        auditability: dict = None,
        redressability: dict = None,
        fairness: dict = None,
        privacy: dict = None,
        sustainability: dict = None,
        load_path: str = None, 
        save_path: str = None
    ):
        self._template = self._load_template()

        if load_path:
            with open(load_path, "r", encoding="utf-8") as f:
                self._factsheet = json.load(f)
        else:
            self._factsheet = copy.deepcopy(self._template)

        # Apply values section by section
        self._apply_section("general", general)
        self._apply_section("governance", governance)
        self._apply_section("auditability", auditability)
        self._apply_section("redressability", redressability)
        self._apply_section("fairness", fairness)
        self._apply_section("privacy", privacy)
        self._apply_section("sustainability", sustainability)

        if save_path:
            self.save(save_path)

    def _load_template(self):
        '''
        Load the default factsheet template.
        '''
        with open(DEFAULT_FACTSHEET_FILE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _apply_section(self, section_name: str, values: dict):
        """
        Safely inject values into a given section.
        """
        if values is None:
            return

        if section_name not in self._factsheet:
            raise ValueError(f"Invalid section: {section_name}")

        for field, value in values.items():
            if field not in self._factsheet[section_name]:
                raise ValueError(f"Invalid field: {section_name}.{field}")

            self._factsheet[section_name][field]["value"] = value

    def to_dict(self):
        return self._factsheet

    def save(self, path="factsheet.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                to_json_safe(self._factsheet),
                f,
                indent=4,
                ensure_ascii=False
            )

    @staticmethod
    def create_factsheet_interactive(output_path="factsheet.json"):
        template = Factsheet()._load_template()
        factsheet = {}

        for section, fields in template.items():
            print(f"\n=== {section.upper()} ===")
            factsheet[section] = {}

            for field, meta in fields.items():
                print(f"\n{field}")
                print(f"  {meta.get('description', '')}")

                current_value = meta.get("value", None)
                if current_value is not None:
                    print(f"  Current value: {current_value}")

                user_input = input("New value (Enter to skip): ").strip()

                if user_input:
                    try:
                        parsed_value = json.loads(user_input)
                    except json.JSONDecodeError:
                        parsed_value = user_input
                else:
                    parsed_value = None

                factsheet[section][field] = {
                    "description": meta.get("description", ""),
                    "value": parsed_value
                }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(factsheet, f, indent=4, ensure_ascii=False)

        print(f"\nFactsheet saved to {output_path}")

        return Factsheet(load_path=output_path)
    
    def __getitem__(self, key):
            """Allows accessing the internal dict using factsheet['key']"""
            return self._factsheet[key]
    
    def get(self, key, default=None):
        """Allows accessing the internal dict using factsheet.get('key')"""
        return self._factsheet.get(key, default)