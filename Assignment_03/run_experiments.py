import os
import subprocess

import torch
import yaml

# Define directories and paths
generated_configs_dir = "./wandb/sweep-p0mqerym"  # Path to generated wandb configs
base_config_path = "./configs/base_config.yaml"
# output_results_dir = "./experiment_results"
# os.makedirs(output_results_dir, exist_ok=True)

# Load the base configuration template
with open(base_config_path, "r") as base_file:
    base_config = yaml.safe_load(base_file)

# Loop over each generated config file
for config_file in os.listdir(generated_configs_dir):
    if config_file.endswith(".yaml"):
        config_path = os.path.join(generated_configs_dir, config_file)

        # Load the generated configuration values
        with open(config_path, "r") as generated_file:
            generated_config = yaml.safe_load(generated_file)

        # Merge generated values into the base config
        experiment_config = base_config.copy()
        experiment_config["training"]["model"] = generated_config["model"]["value"]

        experiment_config["training"]["optimizer"]["name"] = generated_config[
            "optimizer"
        ]["value"]
        experiment_config["training"]["optimizer"]["params"]["lr"] = generated_config[
            "learning_rate"
        ]["value"]

        if experiment_config["training"]["optimizer"]["name"] == "SGD":
            experiment_config["training"]["optimizer"]["params"]["momentum"] = (
                generated_config["momentum"]["value"]
            )
            experiment_config["training"]["optimizer"]["params"]["weight_decay"] = (
                generated_config["weight_decay"]["value"]
            )
            experiment_config["training"]["optimizer"]["params"]["nesterov"] = True
        elif experiment_config["training"]["optimizer"]["name"] == "Adam":
            # remove momentum and weight_decay and nesterov
            experiment_config["training"]["optimizer"]["params"].pop("momentum", None)
            experiment_config["training"]["optimizer"]["params"].pop(
                "weight_decay", None
            )
            experiment_config["training"]["optimizer"]["params"].pop("nesterov", None)

        experiment_config["dataset"]["runtime_transform_script"] = generated_config[
            "runtime_transform_script"
        ]["value"]

        new_config_name = f"experiment_{config_file}"
        new_config_path = os.path.join(
            os.path.dirname(__file__), "configs", new_config_name
        )

        new_config_name_dir = os.path.splitext(new_config_name)[0]

        experiment_config["output"]["save_dir"] = os.path.join(
            os.path.dirname(__file__), "output", new_config_name_dir
        )

        with open(new_config_path, "w") as new_file:
            yaml.dump(experiment_config, new_file)

        # subprocess.run(
        #     ["python", "training.py", "--config", new_config_path],
        #     stdout=subprocess.PIPE,
        #     stderr=subprocess.PIPE,
        # )

        os.system(f"python training.py --config {new_config_path}")

        torch.cuda.empty_cache()
