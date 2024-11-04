import json
import os


def find_singular_reports():
    reports = dict()

    general_output_dir = os.path.join(os.getcwd(), "output")
    run_reports = [
        x
        for x in os.listdir(general_output_dir)
        if os.path.isdir(os.path.join(general_output_dir, x))
        and x.startswith("experiment_config-")
    ]

    for run_report in run_reports:
        run_report_dir = os.path.join(general_output_dir, run_report)
        run_report_file_name = "best_metrics.json"

        run_report_file_path = os.path.join(run_report_dir, run_report_file_name)
        if os.path.exists(run_report_file_path):
            with open(run_report_file_path, "r") as f:
                report = json.load(f)
                config_name = run_report.split("_", maxsplit=1)[1].strip()
                reports[config_name] = report

    return reports


def create_md_table(data):
    table = "| Config | best_train_accuracy | best_test_accuracy | best_loss | best_test_loss |\n"
    table += "| ------ | -------- | --------- | ------ | -- |\n"

    for config, metrics in data.items():
        table += f"| {config} | {metrics['best_train_accuracy']} | {metrics['best_test_accuracy']} | {metrics['best_loss']} | {metrics['best_test_loss']} |\n"

    return table


def main():
    reports = find_singular_reports()
    table = create_md_table(reports)

    with open("report.md", "w") as f:
        f.write(table)


if __name__ == "__main__":
    main()
