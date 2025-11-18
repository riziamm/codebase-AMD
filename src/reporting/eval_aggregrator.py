# Confirmed correct content for 6_eval_process.py

import json
import argparse
from pathlib import Path

def format_classification_report_dict(report_dict):
    if not isinstance(report_dict, dict):
        return "Invalid classification report format (expected dict)."
    output_lines = []
    headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    col_widths = [15, 10, 10, 10, 10]
    header_line = "".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)])
    output_lines.append(header_line)
    output_lines.append("-" * sum(col_widths))
    class_keys = [k for k in report_dict.keys() if k.isdigit()] 
    
    for key in sorted(class_keys):
        metrics = report_dict.get(key, {})
        if isinstance(metrics, dict):
            precision = f"{metrics.get('precision', 0):.2f}"
            recall = f"{metrics.get('recall', 0):.2f}"
            f1_score = f"{metrics.get('f1-score', 0):.2f}"
            support = f"{int(metrics.get('support', 0))}"
            line_items = [key, precision, recall, f1_score, support]
            output_lines.append("".join([f"{item:<{w}}" for item, w in zip(line_items, col_widths)]))
            
    output_lines.append("-" * sum(col_widths))
    
    for avg_key in ['macro avg', 'weighted avg']:
        if avg_key in report_dict and isinstance(report_dict[avg_key], dict):
            metrics = report_dict[avg_key]
            precision = f"{metrics.get('precision', 0):.2f}"
            recall = f"{metrics.get('recall', 0):.2f}"
            f1_score = f"{metrics.get('f1-score', 0):.2f}"
            support = f"{int(metrics.get('support', 0))}"
            line_items = [avg_key, precision, recall, f1_score, support]
            output_lines.append("".join([f"{item:<{w}}" for item, w in zip(line_items, col_widths)]))
            
    return "\n".join(output_lines)

def main():
    parser = argparse.ArgumentParser(description="Aggregate all evaluation metrics for a single experiment.")
    parser.add_argument('--exp_dir', required=True, help="Path to the specific experiment directory to process.")
    args = parser.parse_args()

    exp_path = Path(args.exp_dir)
    if not exp_path.is_dir():
        print(f"Error: Experiment directory not found at {exp_path}")
        return

    metric_files = list(exp_path.rglob('comparison_metrics.json'))
    if not metric_files:
        print(f"Warning: No 'comparison_metrics.json' files found in {exp_path}. Nothing to aggregate.")
        return

    print(f"Found {len(metric_files)} metric files to process for experiment {exp_path.name}.")
    all_results = []

    for metrics_json_path in metric_files:
        print(f"  - Processing {metrics_json_path}")
        try:
            dataset_type = metrics_json_path.parent.name
            with open(metrics_json_path, 'r') as f:
                raw_data = json.load(f)

            eval_results = raw_data.get('evaluation_results', {})
            models_data = eval_results.get('models', {})
            stats_data = eval_results.get('statistical_tests', {})
            
            if not models_data:
                print(f"    Warning: 'models' key is empty or missing in {metrics_json_path}. Skipping.")
                continue
                
            all_results.append({
                'dataset_type': dataset_type,
                'models_data': models_data,
                'stats_data': stats_data
            })
        except Exception as e:
            print(f"    Error processing file {metrics_json_path}: {e}")

    if not all_results:
        print("No results were successfully processed. Exiting.")
        return

    json_output_path = exp_path / f"{exp_path.name}_aggregated_summary.json"
    txt_output_path = exp_path / f"{exp_path.name}_aggregated_summary.txt"

    try:
        with open(json_output_path, 'w') as f:
            json.dump(all_results, f, indent=4)
        print(f"Successfully saved aggregated JSON report to: {json_output_path}")
    except Exception as e:
        print(f"Error saving aggregated JSON: {e}")

    try:
        with open(txt_output_path, 'w') as f:
            f.write(f"Comprehensive Evaluation Summary for Experiment: {exp_path.name}\n")
            f.write("="*80 + "\n")

            for res_block in all_results:
                dataset_type = res_block['dataset_type']
                models_data = res_block['models_data']
                stats_data = res_block.get('stats_data', {})

                f.write(f"\n{'--'*25}\n")
                f.write(f"Dataset: {dataset_type.replace('_', ' ').title()}\n")
                f.write(f"{'--'*25}\n")

                sorted_models = sorted(models_data.items(), key=lambda item: item[1].get('f1_score', 0), reverse=True)

                for model_name, model_metrics in sorted_models:
                    f.write(f"\nModel: {model_name}\n" + "-"*40 + "\n")
                    f.write(f"  - Accuracy:        {model_metrics.get('accuracy', 'N/A'):.4f}\n")
                    f.write(f"  - F1 Score:        {model_metrics.get('f1_score', 'N/A'):.4f}\n")
                    f.write(f"  - ROC AUC:         {model_metrics.get('roc_auc', 'N/A'):.4f}\n")
                    
                    f1_ci = model_metrics.get('f1_score_ci')
                    if f1_ci and isinstance(f1_ci, list) and len(f1_ci) == 2:
                        f.write(f"  - F1 Score CI (95%): [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]\n")

                    f.write("\n  Classification Report:\n")
                    report_str = format_classification_report_dict(model_metrics.get('classification_report', {}))
                    indented_report = "    " + report_str.replace("\n", "\n    ")
                    f.write(indented_report + "\n")

                mcnemar_tests = stats_data.get('mcnemar', {})
                if mcnemar_tests:
                    f.write("\nPairwise Statistical Comparison (McNemar's Test)\n")
                    f.write("(p-value < 0.05 suggests a significant difference in model error rates)\n\n")
                    for test_name, test_results in mcnemar_tests.items():
                        p_value = test_results.get('p_value', -1)
                        significance = "Significant" if p_value != -1 and p_value < 0.05 else "Not Significant"
                        f.write(f"  - {test_name.replace('_', ' ')}: p-value = {p_value:.4f} ({significance})\n")
                f.write("\n")
        print(f"Successfully saved aggregated TXT report to: {txt_output_path}")
    except Exception as e:
        print(f"Error saving aggregated TXT report: {e}")

if __name__ == '__main__':
    main()