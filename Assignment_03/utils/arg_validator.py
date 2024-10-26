import argparse
import os


class ArgValidator:
    @staticmethod
    def parse_and_validate_args():
        """Parses and validates the command-line arguments."""
        parser = argparse.ArgumentParser(
            description="Training Pipeline Argument Parser"
        )
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to the configuration file (YAML format)",
        )

        args = parser.parse_args()

        # Validate the config file path
        config_path = args.config
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"The specified config file '{config_path}' does not exist."
            )

        if not config_path.endswith(".yaml") and not config_path.endswith(".yml"):
            raise ValueError(
                "The config file must be a YAML file with a .yaml or .yml extension."
            )

        return config_path
