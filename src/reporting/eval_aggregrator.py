# Confirmed correct content for 6_eval_process.py

import json
import argparse
from pathlib import Path

def format_classification_report_dict(report_dict):
    # (This helper function remains the same as provided in the previous response)
    if not isinstance(report_dict, dict):
        return "Invalid classification report format (expected dict)."
    output_lines = []
    headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    col_widths = [15, 10, 10, 10, 10]
    header_line = "".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)])
    output_lines.append(header_line)
    output_lines.append("-" * sum(col_widths))
    class_keys = [k for k in report_dict.keys() if k.isdigit()] # Process only numeric class keys
    
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





# import json
# import argparse
# import pandas as pd
# from pathlib import Path
# import numpy as np

# # Helper Function to format Classification Report (from 5_evaluate_test.py) 
# def format_classification_report_dict(report_dict):
#     if not isinstance(report_dict, dict):
#         return "Invalid classification report format (expected dict)."
#     output_lines = [] 
#     headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
#     col_widths = [15, 10, 10, 10, 10]

#     header_line = "".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)])
#     output_lines.append(header_line)
#     output_lines.append("-" * sum(col_widths))

#     class_keys = [k for k in report_dict.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')]
#     for key in sorted(class_keys):
#         metrics = report_dict[key]
#         if isinstance(metrics, dict): # [cite: 6]
#             precision = f"{metrics.get('precision', 0):.2f}"
#             recall = f"{metrics.get('recall', 0):.2f}"
#             f1_score = f"{metrics.get('f1-score', 0):.2f}"
#             support = f"{metrics.get('support', ''):}"
#             line_items = [key, precision, recall, f1_score, support]
#             output_lines.append("".join([f"{item:<{w}}" for item, w in zip(line_items, col_widths)])) # [cite: 7]

#     output_lines.append("-" * sum(col_widths))
#     accuracy_val = report_dict.get('accuracy')
#     if accuracy_val is not None:
#         acc_line = f"{'accuracy':<{col_widths[0]}}" + "".join([' ' * col_widths[i] for i in range(1, 3)]) + f"{accuracy_val:<{col_widths[3]}.2f}"
#         output_lines.append(acc_line)

#     for avg_key in ['macro avg', 'weighted avg']:
#         if avg_key in report_dict and isinstance(report_dict[avg_key], dict): # [cite: 8]
#             metrics = report_dict[avg_key]
#             precision = f"{metrics.get('precision', 0):.2f}"
#             recall = f"{metrics.get('recall', 0):.2f}"
#             f1_score = f"{metrics.get('f1-score', 0):.2f}"
#             support = f"{metrics.get('support', ''):}"
#             line_items = [avg_key, precision, recall, f1_score, support]
#             output_lines.append("".join([f"{item:<{w}}" for item, w in zip(line_items, col_widths)])) 
#     return "\n".join(output_lines)

# def main():
#     parser = argparse.ArgumentParser(description="Process (potentially aggregated) evaluation metrics.")
#     parser.add_argument('--exp_name', required=True, help="Experiment name/identifier.")
#     parser.add_argument('--dataset_type', required=True, choices=['validation', 'holdout'], help="Type of dataset evaluated.")
#     parser.add_argument('--metrics_json_path', required=True, help="Path to the metrics JSON file (e.g., comparison_metrics.json or evaluation_metrics.json).")
#     parser.add_argument('--output_csv', required=True, help="Path to the global CSV file for appending results.")
#     parser.add_argument('--reports_output_dir', required=True, help="Directory to save individual classification report text files.")

#     args = parser.parse_args()
#     all_results_for_json = []

#     metrics_json_path = Path(args.metrics_json_path)
#     output_csv_path = Path(args.output_csv)
#     reports_output_dir = Path(args.reports_output_dir)
#     reports_output_dir.mkdir(parents=True, exist_ok=True)

#     if not metrics_json_path.exists():
#         print(f"Error: Metrics JSON file not found at {metrics_json_path}")
#         # If the main JSON is missing, we can't process models, so we don't add to CSV here.
#         # The shell script should handle the case where main.py fails to produce this JSON.
#         return

#     with open(metrics_json_path, 'r') as f:
#         try:
#             raw_metrics_data = json.load(f)
#         except json.JSONDecodeError as e:
#             print(f"Error decoding JSON from {metrics_json_path}: {e}")
#             return

#     # The JSON could be from `evaluate_saved_model` (single model) or `compare_models` (multiple models)
#     # `compare_models` returns a dict where keys are model names.
#     # `evaluate_saved_model` returns a dict of metrics directly.
#     # `main.py` saves the `results` dict. If `args.model_paths` was used, `results` is from `compare_models`.
#     # If `args.model_path` (single) was used, `results` is from `evaluate_saved_model`.

#     models_data = {}
#     # Check if the JSON is from `compare_models` or `evaluate_saved_model`
    
#     # if 'accuracy' in raw_metrics_data and 'f1_score' in raw_metrics_data: # Likely single model metrics
#     #     # This case should ideally not happen if the shell script always calls with all model_paths for aggregation.
#     #     # For robustness, we handle it by wrapping it.
#     #     # The model name would be unknown from this JSON alone, might need to pass it or infer.
#     #     # For now, if this happens, the shell script logic might need adjustment for model_name.
#     #     # Let's assume the shell script passes a JSON from compare_models.
#     #     print(f"Warning: Metrics JSON {metrics_json_path} appears to be for a single model. Aggregation works best with multi-model 'comparison_metrics.json'.")
#     #     # Try to infer model name if possible or use a placeholder
#     #     inferred_model_name = raw_metrics_data.get('model_name', 'unknown_single_model') # if 'model_name' key exists
#     #     models_data = {inferred_model_name: raw_metrics_data}
#     # elif isinstance(raw_metrics_data, dict): # Likely a dict of models (output of compare_models)
#     #     models_data = raw_metrics_data
#     # else:
#     #     print(f"Error: Unexpected format in metrics JSON {metrics_json_path}")
#     #     return
    
#     try:
#     # This is the output of 'compare_models'
#         models_data = raw_metrics_data['models']
#         statistical_data = raw_metrics_data.get('statistical_tests', {})
#     except KeyError:
#         # This handles the case where the JSON is from 'evaluate_saved_model'
#         # which has a flatter structure.
#         print(f"Warning: Did not find 'models' key. Assuming single-model evaluation format.")
#         # logic for handling single models 
#         if 'accuracy' in raw_metrics_data and 'f1_score' in raw_metrics_data: # Likely single model metrics
#                 # Assume the shell script passes a JSON from compare_models.
#             print(f"Warning: Metrics JSON {metrics_json_path} appears to be for a single model. Aggregation works best with multi-model 'comparison_metrics.json'.")
#             # Try to infer model name if possible or use a placeholder
#             inferred_model_name = raw_metrics_data.get('model_name', 'unknown_single_model') # if 'model_name' key exists
#             models_data = {inferred_model_name: raw_metrics_data}
#         elif isinstance(raw_metrics_data, dict): # Likely a dict of models (output of compare_models)
#             models_data = raw_metrics_data
#         else:
#             print(f"Error: Unexpected format in metrics JSON {metrics_json_path}")
#             return
#         #  multi-model case from 'compare_models'.
#         return

#     # --- Create a comprehensive text report (classification reports and statistical tests)
#     report_file_path = reports_output_dir / f"evaluation_summary_{args.dataset_type}.txt"
#     with open(report_file_path, 'w') as f:
#         f.write(f"Comprehensive Evaluation Report\n")
#         f.write(f"Experiment: {args.exp_name}\n")
#         f.write(f"Dataset: {args.dataset_type}\n")
#         f.write("="*70 + "\n\n")

#         # 1. Write the classification report for each model
#         for model_name, model_metrics in models_data.items():
#             f.write(f"--- Model: {model_name} ---\n")
#             report_str = model_metrics.get('classification_report_str', 'Classification report not available.')
#             f.write(report_str)
#             f.write("\n" + "-"*70 + "\n\n")

#         # 2. Write the statistical test results
#         mcnemar_tests = statistical_data.get('mcnemar', {})
#         if mcnemar_tests:
#             f.write("Pairwise Statistical Comparison (McNemar's Test)\n")
#             f.write("(p-value < 0.05 suggests a significant difference in error rates)\n\n")
#             for test_name, test_results in mcnemar_tests.items():
#                 p_value = test_results.get('p_value', -1)
#                 significance = "Significant" if p_value < 0.05 else "Not Significant"
#                 f.write(f"  - {test_name.replace('_', ' ')}: p-value = {p_value:.4f} ({significance})\n")
#         else:
#             f.write("No McNemar's test data found.\n")

#     print(f"Saved comprehensive text report to {report_file_path}")
#     # ---  ---    




#     rows_to_append = []

#     for model_name, model_metrics in models_data.items():
#         accuracy = model_metrics.get('accuracy', 'N/A')
#         f1_score_weighted = model_metrics.get('f1_score', 'N/A') # This is typically weighted F1
#         auc = model_metrics.get('auc', 'N/A')
        

#         precision_weighted = 'N/A'
#         recall_weighted = 'N/A'
#         # Classification report can be a dict or a string.
#         # `compare_models` stores `classification_report` (dict) and `classification_report_str`
#         classification_report_obj = model_metrics.get('classification_report', {}) # dict from compare_models
#         formatted_report_str = ""

#         if isinstance(classification_report_obj, dict):
#             if 'weighted avg' in classification_report_obj:
#                 precision_weighted = classification_report_obj['weighted avg'].get('precision', 'N/A')
#                 recall_weighted = classification_report_obj['weighted avg'].get('recall', 'N/A')
#             formatted_report_str = format_classification_report_dict(classification_report_obj)
#         elif isinstance(model_metrics.get('classification_report_str'), str): # If only string is available
#              formatted_report_str = model_metrics['classification_report_str']
#              # Attempt to parse string report for weighted averages if dict not available (more complex)
#         else:
#             formatted_report_str = "Classification report not available or in unexpected format."


#         auc = model_metrics.get('roc_auc', 'N/A') # Assuming 'auc'/'roc_auc' 

#         class_report_file = reports_output_dir / f"{model_name}_{args.dataset_type}_report.txt"
#         with open(class_report_file, 'w') as f:
#             f.write(formatted_report_str)
#         print(f"Saved classification report for {model_name} ({args.dataset_type}) to {class_report_file}")

#         # Extract new CI data with a default value
#         # accuracy_ci = model_metrics.get('accuracy_ci', ['N/A', 'N/A'])
#         f1_ci = model_metrics.get('f1_score_ci', ['N/A', 'N/A'])

#         data_row = {
#             'experiment_name': args.exp_name,
#             'model_name': model_name,
#             'dataset_type': args.dataset_type,
#             'F1_Score_Weighted': f"{f1_score_weighted:.4f}" if isinstance(f1_score_weighted, float) else f1_score_weighted,
#             'Accuracy': f"{accuracy:.4f}" if isinstance(accuracy, float) else accuracy,
#             'Precision_Weighted': f"{precision_weighted:.4f}" if isinstance(precision_weighted, float) else precision_weighted,
#             'Recall_Weighted': f"{recall_weighted:.4f}" if isinstance(recall_weighted, float) else recall_weighted,
#             # 'AUC': f"{auc:.4f}" if isinstance(auc, float) else auc,
#             # --- NEW COLUMNS ---
#             'AUC': f"{model_metrics.get('roc_auc', 'N/A'):.4f}" 
#             # 'Accuracy_CI_Lower': f"{accuracy_ci[0]:.4f}" if isinstance(accuracy_ci[0], float) else 'N/A',
#             # 'Accuracy_CI_Upper': f"{accuracy_ci[1]:.4f}" if isinstance(accuracy_ci[1], float) else 'N/A'
#             'F1_CI_Lower': f"{f1_ci[0]:.4f}" if isinstance(f1_ci[0], float) else 'N/A',
#             'F1_CI_Upper': f"{f1_ci[1]:.4f}" if isinstance(f1_ci[1], float) else 'N/A'
#         }
#         all_results_for_json.append(data_row)
#         rows_to_append.append(data_row)

#     if rows_to_append:
#         df_to_append = pd.DataFrame(rows_to_append)
#         if output_csv_path.exists():
#             df_to_append.to_csv(output_csv_path, mode='a', header=False, index=False)
#         else:
#             df_to_append.to_csv(output_csv_path, mode='w', header=True, index=False)
#         print(f"Appended {len(rows_to_append)} model(s) metrics for {args.exp_name} ({args.dataset_type}) to CSV: {output_csv_path}")
#     else:
#         print(f"No model data found to process in {metrics_json_path} for {args.exp_name} ({args.dataset_type}).")
        
#     # --- NEW: Write the statistical summary file ---
#     if statistical_data:
#         stats_report_file = reports_output_dir / f"statistical_summary_{args.dataset_type}.txt"
#         with open(stats_report_file, 'w') as f:
#             f.write(f"Statistical Comparison Report\n")
#             f.write(f"Experiment: {args.exp_name}\nDataset: {args.dataset_type}\n")
#             f.write("="*50 + "\n\n")
            
#             mcnemar_tests = statistical_data.get('mcnemar', {})
#             if mcnemar_tests:
#                 f.write("McNemar's Test Results (p < 0.05 suggests a significant difference in error rates):\n")
#                 for test_name, test_results in mcnemar_tests.items():
#                     p_value = test_results.get('p_value', -1)
#                     significance = "Significant" if p_value < 0.05 else "Not Significant"
#                     f.write(f"  - {test_name.replace('_', ' ')}: p-value = {p_value:.4f} ({significance})\n")
#             else:
#                 f.write("No McNemar's test data found.\n")

#         print(f"Saved statistical summary to {stats_report_file}")
        
#     if all_results_for_json:
#         aggregated_json_path = Path(args.reports_output_dir).parent / 'aggregated_evaluation_summary.json'
#         try:
#             with open(aggregated_json_path, 'w') as f:
#                 json.dump(all_results_for_json, f, indent=4)
#             print(f"Saved aggregated JSON report to: {aggregated_json_path}")
#         except Exception as e:
#             print(f"Error saving aggregated JSON: {e}")

# if __name__ == '__main__':
#     main()