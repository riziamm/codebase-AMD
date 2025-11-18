import os
import json
import pandas as pd
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, Template 
from datetime import datetime
import traceback 
import base64 

#  Helper function to safely get nested dictionary values 
def safe_get(data, keys, default=None):
    """Safely retrieve nested dictionary values."""
    if not isinstance(data, dict):
        return default
    temp = data
    for key in keys:
        if isinstance(temp, dict) and key in temp:
            temp = temp[key]
        else:
            return default
    if not isinstance(temp, (str, int, float, bool, list, dict, type(None))):
        return str(temp)
    return temp

#  Helper Function to format Classification Report 
def format_classification_report_dict(report_dict):
    """Formats a classification report dictionary into a preformatted string."""
    if not isinstance(report_dict, dict):
        return "Invalid classification report format (expected dict)."

    output_lines = []
    headers = ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']
    col_widths = [15, 10, 10, 10, 10] # Adjust widths as needed

    # Header line
    header_line = "".join([f"{h:<{w}}" for h, w in zip(headers, col_widths)])
    output_lines.append(header_line)
    output_lines.append("-" * sum(col_widths))

    # Class metrics
    class_keys = [k for k in report_dict.keys() if k not in ('accuracy', 'macro avg', 'weighted avg')]
    for key in sorted(class_keys):
        metrics = report_dict[key]
        if isinstance(metrics, dict):
             precision = f"{metrics.get('precision', 0):.2f}"
             recall = f"{metrics.get('recall', 0):.2f}"
             f1_score = f"{metrics.get('f1-score', 0):.2f}"
             support = f"{metrics.get('support', ''):}" # Support might be float or int
             line_items = [key, precision, recall, f1_score, support]
             output_lines.append("".join([f"{item:<{w}}" for item, w in zip(line_items, col_widths)]))

    output_lines.append("-" * sum(col_widths))

    # Summary metrics
    accuracy = report_dict.get('accuracy')
    if accuracy is not None:
        acc_line = f"{'accuracy':<{col_widths[0]}}" + "".join([' ' * col_widths[i] for i in range(1, 3)]) + f"{accuracy:<{col_widths[3]}.2f}"
        output_lines.append(acc_line)

    for avg_key in ['macro avg', 'weighted avg']:
        if avg_key in report_dict and isinstance(report_dict[avg_key], dict):
            metrics = report_dict[avg_key]
            precision = f"{metrics.get('precision', 0):.2f}"
            recall = f"{metrics.get('recall', 0):.2f}"
            f1_score = f"{metrics.get('f1-score', 0):.2f}"
            support = f"{metrics.get('support', ''):}"
            line_items = [avg_key, precision, recall, f1_score, support]
            output_lines.append("".join([f"{item:<{w}}" for item, w in zip(line_items, col_widths)]))

    return "\n".join(output_lines)


def generate_detailed_batch_report(batch_dir_path_str):
    """
    Generates a detailed HTML report aggregating results from multiple
    experiments within a batch directory.

    Args:
        batch_dir_path_str: Path string to the main batch directory
                           (e.g., 'reports/batch_20250409_103000').
    """
    batch_dir = Path(batch_dir_path_str)
    if not batch_dir.is_dir():
        print(f"Error: Batch directory not found: {batch_dir}")
        return

    print(f"Starting detailed batch report generation for: {batch_dir}")
    all_experiments_data = [] 

    experiment_dirs = sorted([d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith('experiment_')])

    if not experiment_dirs:
        print(f"Warning: No experiment directories found in {batch_dir}. Cannot generate report.")
        return

    print(f"Found {len(experiment_dirs)} experiment directories.")

    for exp_dir in experiment_dirs:
        print(f"\nProcessing experiment: {exp_dir.name}")
        exp_data = {
            'experiment_name': exp_dir.name,
            'config': "N/A",
            'metrics': [], 
            'model_reports': {}, 
            'overview_figures': [],
            'model_figures': {}, 
            'general_shap_figures': [],
            'best_model_name': None
        }

        config_path = exp_dir / 'experiment_config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                exp_data['config'] = json.dumps(config_data, indent=4)
                print(f"  Loaded config from {config_path}")
            except Exception as e:
                print(f"  Warning: Could not load/parse config {config_path}: {e}")
                exp_data['config'] = f"Error loading config: {e}"
        else:
            print(f"  Warning: Config file not found: {config_path}")
            exp_data['config'] = "Config file not found."

        metrics_dir = exp_dir / 'metrics'
        model_names_found = set()

        consolidated_metrics_path = metrics_dir / 'all_model_metrics.json'
        if consolidated_metrics_path.exists():
            print(f"  Found consolidated metrics: {consolidated_metrics_path}")
            try:
                with open(consolidated_metrics_path, 'r') as f:
                    metrics_data = json.load(f)
                exp_data['best_model_name'] = metrics_data.get('best_model_name')
                all_models_metrics = metrics_data.get('models', {})

                for model_name, model_results in all_models_metrics.items():
                    model_names_found.add(model_name)
                    exp_data['metrics'].append({
                        'model_name': model_name,
                        'accuracy': model_results.get('accuracy'),
                        'f1_score': model_results.get('f1_score'),
                        'precision': model_results.get('precision'),
                        'recall': model_results.get('recall')
                    })
                    if 'classification_report' in model_results:
                        exp_data['model_reports'][model_name] = model_results['classification_report']

                if exp_data['best_model_name'] not in all_models_metrics:
                    print(f"  Warning: Best model name '{exp_data['best_model_name']}' not found in consolidated metrics models. Will determine best by F1.")
                    exp_data['best_model_name'] = None

            except Exception as e:
                print(f"  Warning: Could not load/parse consolidated metrics {consolidated_metrics_path}: {e}")
                exp_data['best_model_name'] = None

        individual_metrics_files = list(metrics_dir.glob('*_metrics.json'))
        if not exp_data['metrics'] and individual_metrics_files:
             print(f"  Consolidated metrics missing or failed. Loading individual metric files.")
             for metric_file in individual_metrics_files:
                  try:
                      with open(metric_file, 'r') as f:
                          metric_data = json.load(f)
                      model_name = metric_data.get('model_name', metric_file.stem.replace('_metrics',''))
                      if model_name not in model_names_found:
                           model_names_found.add(model_name)
                           exp_data['metrics'].append({
                               'model_name': model_name,
                               'accuracy': metric_data.get('accuracy'),
                               'f1_score': metric_data.get('f1_score'),
                               'precision': metric_data.get('precision'),
                               'recall': metric_data.get('recall')
                           })
                      if model_name not in exp_data['model_reports']:
                           report_file = metrics_dir / f"{model_name}_classification_report.json"
                           if report_file.exists():
                                with open(report_file, 'r') as f:
                                    exp_data['model_reports'][model_name] = json.load(f)
                  except Exception as e:
                      print(f"  Warning: Could not load/parse individual metric file {metric_file}: {e}")

        if not exp_data['best_model_name'] and exp_data['metrics']:
            exp_data['metrics'].sort(key=lambda x: x.get('f1_score', -1) if x.get('f1_score') is not None else -1, reverse=True)
            if exp_data['metrics'][0].get('f1_score') is not None:
                 exp_data['best_model_name'] = exp_data['metrics'][0]['model_name']
                 print(f"  Determined best model by F1 score: {exp_data['best_model_name']}")

        exp_data['metrics'].sort(key=lambda x: x.get('f1_score', -1) if x.get('f1_score') is not None else -1, reverse=True)
        model_names_in_order = [m['model_name'] for m in exp_data['metrics']]
        print(f"  Loaded metrics for models: {model_names_in_order}")

        figures_dir = exp_dir / 'figures'
        if figures_dir.is_dir():
            all_figure_files = list(figures_dir.glob('*.png'))
            print(f"  Found {len(all_figure_files)} figures in {figures_dir}")

            for model_name in model_names_in_order:
                exp_data['model_figures'][model_name] = {'plots': [], 'shap': []}

            for fig_path in all_figure_files:
                try:
                    with open(fig_path, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode()
                    fig_name = fig_path.stem
                    fig_name_lower = fig_name.lower()
                    fig_data_dict = {'name': fig_name, 'data': img_data}

                    assigned = False

                    if any(term in fig_name_lower for term in ["class_distribution", "model_comparison"]):
                        exp_data['overview_figures'].append(fig_data_dict)
                        print(f"    Categorized as Overview: {fig_name}")
                        assigned = True
                        continue

                    is_shap = any(term in fig_name_lower for term in ['shap', 'waterfall', 'force', 'beeswarm', 'feature_importance', 'importance', 'coefficient', 'feature_gp_importance'])
                    if is_shap:
                         model_assigned = False
                         for model_name in model_names_in_order:
                             model_variants = [ model_name.lower(), model_name.replace(' ', '_').lower(), model_name.replace(' ', '').lower() ]
                             if any(variant in fig_name_lower for variant in model_variants):
                                 if model_name in exp_data['model_figures']:
                                      exp_data['model_figures'][model_name]['shap'].append(fig_data_dict)
                                      print(f"    Categorized as SHAP for {model_name}: {fig_name}")
                                      model_assigned = True
                                      assigned = True
                                      break
                         if not model_assigned:
                              exp_data['general_shap_figures'].append(fig_data_dict)
                              print(f"    Categorized as General SHAP: {fig_name}")
                              assigned = True
                         continue

                    if not assigned:
                        for model_name in model_names_in_order:
                            model_variants = [ model_name.lower(), model_name.replace(' ', '_').lower(), model_name.replace(' ', '').lower() ]
                            if any(variant in fig_name_lower for variant in model_variants):
                                if model_name in exp_data['model_figures']:
                                     exp_data['model_figures'][model_name]['plots'].append(fig_data_dict)
                                     print(f"    Categorized as Plot for {model_name}: {fig_name}")
                                     assigned = True
                                     break

                    if not assigned:
                        print(f"    Warning: Could not categorize figure: {fig_name}. Adding to Overview.")
                        exp_data['overview_figures'].append(fig_data_dict)

                except Exception as e:
                    print(f"    Error processing figure {fig_path}: {e}")
        else:
            print(f"  Warning: Figures directory not found: {figures_dir}")

        all_experiments_data.append(exp_data)
        print("-" * 20)


    #  HTML Template 
    detailed_batch_template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detailed Batch Experiment Report: {{ batch_name }}</title>
    <style>
        body { font-family: sans-serif; margin: 20px; line-height: 1.6; background-color: #f8f9fa; color: #212529; }
        h1 { color: #0056b3; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; margin-bottom: 30px; }
        h2 { color: #0056b3; margin-top: 40px; border-bottom: 1px solid #ced4da; padding-bottom: 8px; }
        h3 { color: #343a40; margin-top: 30px; }
        h4 { color: #495057; margin-top: 25px; }
        .experiment-block { margin-bottom: 50px; padding: 25px; border: 1px solid #dee2e6; border-radius: 8px; background-color: #ffffff; box-shadow: 0 4px 8px rgba(0,0,0,0.05); }
        .section { margin-bottom: 30px; padding-bottom: 20px; border-bottom: 1px dashed #e0e0e0; }
        .section:last-child { border-bottom: none; }
        .figure-container { display: flex; flex-wrap: wrap; justify-content: start; gap: 25px; margin-top: 15px; padding-left: 10px;}
        .figure { text-align: center; background-color: #f8f9fa; padding: 10px; border: 1px solid #dee2e6; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); flex: 1 1 300px; /* Flex behavior */}
        .figure img { max-width: 100%; height: auto; max-height: 450px; border-radius: 3px; display: block; margin: 0 auto;}
        .figure-caption { font-size: 0.9em; color: #495057; margin-top: 8px; word-wrap: break-word; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; background-color: #ffffff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        th, td { border: 1px solid #dee2e6; padding: 10px 12px; text-align: left; }
        th { background-color: #e9ecef; font-weight: bold; }
        tr:nth-child(even) { background-color: #f8f9fa; }
        tr:hover { background-color: #e2e6ea; }
        .config-pre { font-family: monospace; white-space: pre-wrap; padding: 15px; background-color: #e9ecef; border: 1px solid #ced4da; border-radius: 5px; overflow-x: auto; max-height: 300px; }
        .model-report-pre { font-family: monospace; white-space: pre-wrap; padding: 10px; background-color: #f1f3f5; border: 1px solid #ced4da; border-radius: 4px; margin: 10px 0; overflow-x: auto; font-size: 0.85em; }
        .best-model { background-color: rgba(25, 135, 84, 0.1); border-left: 5px solid #198754;} /* Highlight best model rows */
        .best-model-card { border: 2px solid #198754; }
        .summary-box { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 4px solid #0056b3; }
        .metric-box { display: inline-block; margin-right: 15px; text-align: center; min-width: 100px; }
        .metric-value { font-size: 1.2em; font-weight: bold; color: #0056b3; }
        .metric-label { font-size: 0.9em; color: #6c757d; }
        .model-card { margin-bottom: 30px; padding: 15px; border: 1px solid #dee2e6; border-radius: 5px; background-color: #fdfdfd; }
        .na-value { color: #6c757d; font-style: italic; }
        .timestamp { margin-top: 30px; font-size: 0.8em; color: #6c757d; text-align: right; }
    </style>
</head>
<body>
    <h1>Detailed Batch Experiment Report</h1>
    <h2>Batch: {{ batch_name }}</h2>
    <p>Report generated on: {{ generation_time }}</p>

    {% if all_experiments_data %}
        {% for experiment in all_experiments_data %}
        <div class="experiment-block">
            <h2>Experiment: {{ experiment.experiment_name }}</h2>

            {# 1. Experiment Overview #}
            <div class="section">
                <h3>1. Experiment Overview</h3>
                <div class="summary-box">
                    <h4>Best Model: <span style="font-weight:bold; color:#198754;">{{ experiment.best_model_name if experiment.best_model_name else 'N/A' }}</span></h4>

                    {#  CORRECTED SECTION to find and display best_metrics  #}
                    {% set best_metrics = experiment.metrics | selectattr('model_name', 'equalto', experiment.best_model_name) | first %}

                    {% if best_metrics %}
                        {# Display metrics if found #}
                        <div class="metric-box"><span class="metric-label">Accuracy:</span> <span class="metric-value">{{ "%.4f"|format(best_metrics.accuracy) if best_metrics.accuracy is not none else 'N/A' }}</span></div>
                        <div class="metric-box"><span class="metric-label">F1 Score:</span> <span class="metric-value">{{ "%.4f"|format(best_metrics.f1_score) if best_metrics.f1_score is not none else 'N/A' }}</span></div>
                        <div class="metric-box"><span class="metric-label">Precision:</span> <span class="metric-value">{{ "%.4f"|format(best_metrics.precision) if best_metrics.precision is not none else 'N/A' }}</span></div>
                        <div class="metric-box"><span class="metric-label">Recall:</span> <span class="metric-value">{{ "%.4f"|format(best_metrics.recall) if best_metrics.recall is not none else 'N/A' }}</span></div>
                    {% else %}
                         {# Handle cases where metrics weren't found #}
                         {% if experiment.best_model_name and experiment.best_model_name != 'N/A' %}
                             <p>Metrics for best model ({{ experiment.best_model_name }}) not found in the metrics list.</p>
                         {% elif not experiment.metrics %}
                             <p>No metrics available for this experiment.</p>
                         {% else %}
                             <p>Best model name not determined for this experiment.</p>
                         {% endif %}
                    {% endif %}
                    {#  END CORRECTION  #}
                </div>
                <h4>Configuration</h4>
                <pre class="config-pre">{{ experiment.config }}</pre>
            </div>

            {# 2. Data Analysis - Visualizing Class Distribution Plots #}
            <div class="section">
                <h3>2. Data Analysis</h3>
                 <div class="figure-container">
                    {# Filter for class distribution plots specifically #}
                    {% set dist_plots_found = false %} {# Flag #}
                    {% for figure in experiment.overview_figures %}
                         {% if 'class_distribution' in figure.name.lower() %}
                            <div class="figure">
                                <img src="data:image/png;base64,{{ figure.data }}" alt="{{ figure.name }}">
                                <p class="figure-caption">{{ figure.name }}</p>
                            </div>
                             {% set dist_plots_found = true %} {# Set flag #}
                         {% endif %}
                    {% endfor %}
                    {% if not dist_plots_found %} {# Check flag #}
                        <p>No class distribution plots found.</p>
                    {% endif %}
                 </div>
            </div>

            {# 3. Performance Summary #}
            <div class="section">
                <h3>3. Performance Summary</h3>
                <h4>Model Metrics Comparison</h4>
                 <table>
                    <thead>
                        <tr>
                            <th>Model</th><th>Accuracy</th><th>F1 Score</th><th>Precision</th><th>Recall</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for metric in experiment.metrics %}
                        <tr {% if metric.model_name == experiment.best_model_name %}class="best-model"{% endif %}>
                            <td>{{ metric.model_name }}</td>
                            <td>{{ "%.4f"|format(metric.accuracy) if metric.accuracy is not none else 'N/A' }}</td>
                            <td>{{ "%.4f"|format(metric.f1_score) if metric.f1_score is not none else 'N/A' }}</td>
                            <td>{{ "%.4f"|format(metric.precision) if metric.precision is not none else 'N/A' }}</td>
                            <td>{{ "%.4f"|format(metric.recall) if metric.recall is not none else 'N/A' }}</td>
                        </tr>
                        {% else %}
                        <tr><td colspan="5">No model metrics found for this experiment.</td></tr>
                        {% endfor %}
                    </tbody>
                 </table>
                 <h4>Model Comparison Plot</h4>
                 <div class="figure-container">
                    {#  CORRECTED WAY to find model comparison plots  #}
                    {% set found_comparison_plot = false %} {# Flag to check if any were found #}
                    {% for figure in experiment.overview_figures %}
                        {# Check if 'model_comparison' is in the figure name (case-insensitive) #}
                        {% if 'model_comparison' in figure.name.lower() %}
                            <div class="figure">
                                <img src="data:image/png;base64,{{ figure.data }}" alt="{{ figure.name }}">
                                <p class="figure-caption">{{ figure.name }}</p>
                            </div>
                            {% set found_comparison_plot = true %} {# Set flag #}
                        {% endif %}
                    {% endfor %}
                    {# Display message only if no plots were found during the loop #}
                    {% if not found_comparison_plot %}
                         <p>Model comparison plot not found.</p>
                    {% endif %}
                    {#  END CORRECTION  #}
                 </div>
            </div>

            {# 4. Model Performances #}
            <div class="section">
                <h3>4. Model Performances</h3>
                {% for metric in experiment.metrics %}
                    <div class="model-card {% if metric.model_name == experiment.best_model_name %}best-model-card{% endif %}">
                        <h4>{{ metric.model_name }} {% if metric.model_name == experiment.best_model_name %}(Best Model){% endif %}</h4>
                         <div class="metric-box"><span class="metric-label">Accuracy:</span> <span class="metric-value">{{ "%.4f"|format(metric.accuracy) if metric.accuracy is not none else 'N/A' }}</span></div>
                         <div class="metric-box"><span class="metric-label">F1 Score:</span> <span class="metric-value">{{ "%.4f"|format(metric.f1_score) if metric.f1_score is not none else 'N/A' }}</span></div>
                         <div class="metric-box"><span class="metric-label">Precision:</span> <span class="metric-value">{{ "%.4f"|format(metric.precision) if metric.precision is not none else 'N/A' }}</span></div>
                         <div class="metric-box"><span class="metric-label">Recall:</span> <span class="metric-value">{{ "%.4f"|format(metric.recall) if metric.recall is not none else 'N/A' }}</span></div>

                         {% if metric.model_name in experiment.model_reports %}
                             <h5>Classification Report</h5>
                             <pre class="model-report-pre">{{ format_classification_report(experiment.model_reports[metric.model_name]) }}</pre>
                         {% endif %}

                         <h5>Visualizations</h5>
                         <div class="figure-container">
                             {% set model_plots = experiment.model_figures.get(metric.model_name, {}).get('plots', []) %}
                             {% if model_plots %}
                                 {% for figure in model_plots %}
                                     <div class="figure">
                                         <img src="data:image/png;base64,{{ figure.data }}" alt="{{ figure.name }}">
                                         <p class="figure-caption">{{ figure.name }}</p>
                                     </div>
                                 {% endfor %}
                             {% else %}
                                 <p>No specific plots found for this model.</p>
                             {% endif %}
                         </div>
                    </div>
                {% else %}
                     <p>No models found for this experiment.</p>
                {% endfor %}
            </div>

            {# 5. Shapley Analysis #}
            <div class="section">
                 <h3>5. Shapley (SHAP) Analysis</h3>
                 {% if experiment.general_shap_figures %}
                    <h4>General SHAP Plots</h4>
                    <div class="figure-container">
                         {% for figure in experiment.general_shap_figures %}
                             <div class="figure">
                                 <img src="data:image/png;base64,{{ figure.data }}" alt="{{ figure.name }}">
                                 <p class="figure-caption">{{ figure.name }}</p>
                             </div>
                         {% endfor %}
                    </div>
                 {% endif %}

                 {% set model_shap_found = false %}
                 {% for model_name, figures in experiment.model_figures.items() %}
                     {% if figures.shap %}
                         {% set model_shap_found = true %}
                         <h4>SHAP Plots for {{ model_name }}</h4>
                         <div class="figure-container">
                             {% for figure in figures.shap %}
                                 <div class="figure">
                                     <img src="data:image/png;base64,{{ figure.data }}" alt="{{ figure.name }}">
                                     <p class="figure-caption">{{ figure.name }}</p>
                                 </div>
                             {% endfor %}
                         </div>
                     {% endif %}
                 {% else %} {# This else belongs to the for model_name loop #}
                    {# If the loop completes without finding any models with SHAP plots #}
                    {% if not model_shap_found and not experiment.general_shap_figures %}
                         {# This check needs to be done after the loop or differently #}
                    {% endif %}
                 {% endfor %}

                 {# Check if NO shap plots were found at all, after the loop #}
                 {% if not experiment.general_shap_figures and not model_shap_found %}
                      <p>No SHAP analysis plots found for this experiment.</p>
                 {% endif %}
            </div>

        </div> {# End experiment-block #}
        {% endfor %}
    {% else %}
        <p>No experiment data could be processed for this batch.</p>
    {% endif %}

    <div class="timestamp">End of Report</div>
</body>
</html>
"""

    try:
        template = Template(detailed_batch_template_str)

        generation_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html_content = template.render(
            batch_name=batch_dir.name,
            all_experiments_data=all_experiments_data, 
            generation_time=generation_time_str,
            format_classification_report=format_classification_report_dict
        )

        report_path = batch_dir / 'batch_summary_detailed.html' # Use a different name
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"\nSuccessfully generated DETAILED batch summary report: {report_path}")

    except Exception as e:
        print(f"Error generating detailed batch HTML report: {e}")
        print(traceback.format_exc())



if __name__ == "__main__":
    target_batch_directory = "reports/batch_experiments/batch_20250409_202555" # Update path

    if Path(target_batch_directory).is_dir():
        generate_detailed_batch_report(target_batch_directory)
    else:
        print(f"Error: The specified directory does not exist: {target_batch_directory}")
        print("Please provide the correct path to the batch timestamp directory.")