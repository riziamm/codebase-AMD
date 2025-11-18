import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from datetime import datetime
import itertools
from sklearn.model_selection import ParameterGrid
import time
import sys
from collections import defaultdict
from sklearn.metrics import f1_score
# import pip
# pip.main(['install','seaborn'])
# import seaborn as sns
import traceback 
from .training_pipeline import run_classification_pipeline_with_reporting
from .reporting.utils import create_report_directory, save_experiment_config, save_model_metrics


def plot_experiment_results_v1(summary_df, batch_dir):
    """
    Create visualizations of experiment results
    
    Args:
        summary_df: DataFrame with experiment results
        batch_dir: Batch directory path
    """
    # Check if there are successful experiments
    if 'f1_score' not in summary_df.columns or summary_df['f1_score'].isnull().all():
        print("No successful experiments to visualize")
        return
    
    # Drop rows with errors (no f1_score)
    df = summary_df.dropna(subset=['f1_score'])
    
    # Check if we have any valid data to plot
    if len(df) == 0:
        print("No experiments with valid F1 scores to visualize")
        return
    
    # Create figures directory
    figures_dir = Path(batch_dir) / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    try:
        # 1. Bar chart of F1 scores
        plt.figure(figsize=(12, 6))
        
        # Sort by F1 score for better visualization
        df_sorted = df.sort_values('f1_score', ascending=False)
        
        # Check if experiment_name column exists, if not use experiment_id
        x_axis_col = 'experiment_name' if 'experiment_name' in df_sorted.columns else 'experiment_id'
        if x_axis_col not in df_sorted.columns:
            # If neither column exists, create a simple index
            df_sorted['plot_index'] = [f"Exp {i+1}" for i in range(len(df_sorted))]
            x_axis_col = 'plot_index'
            
        plt.bar(df_sorted[x_axis_col], df_sorted['f1_score'])
        plt.xlabel('Experiment')
        plt.ylabel('F1 Score')
        plt.title('F1 Scores Across Experiments')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(figures_dir / 'f1_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Execution times (if available)
        if 'execution_time' in df_sorted.columns and not df_sorted['execution_time'].isnull().all():
            plt.figure(figsize=(12, 6))
            plt.bar(df_sorted[x_axis_col], df_sorted['execution_time'] / 60)  # Convert to minutes
            plt.xlabel('Experiment')
            plt.ylabel('Execution Time (minutes)')
            plt.title('Execution Times Across Experiments')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(figures_dir / 'execution_times.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Parameter influence analysis (if enough experiments)
        if len(df) > 3:  # Lower threshold for analysis
            # Find parameters that vary across experiments
            param_columns = [col for col in df.columns 
                             if col not in ['experiment_id', 'experiment_name', 'best_model_name', 
                                            'accuracy', 'f1_score', 'execution_time', 'report_dir',
                                            'status', 'error_message']]
            
            varying_params = []
            for param in param_columns:
                if param in df.columns and len(df[param].dropna().unique()) > 1:
                    varying_params.append(param)
            
            if varying_params:
                # Plot the influence of each varying parameter on F1 score
                for param in varying_params:
                    try:
                        plt.figure(figsize=(10, 6))
                        
                        # Convert parameter values to string to ensure they can be used as categories
                        df[f"{param}_str"] = df[param].astype(str)
                        
                        # Group by parameter and calculate mean F1 score
                        param_impact = df.groupby(f"{param}_str")['f1_score'].mean().reset_index()
                        param_impact = param_impact.sort_values('f1_score', ascending=False)
                        
                        # Plot
                        plt.bar(param_impact[f"{param}_str"], param_impact['f1_score'])
                        plt.xlabel(param)
                        plt.ylabel('Mean F1 Score')
                        plt.title(f'Impact of {param} on F1 Score')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        plt.savefig(figures_dir / f'param_impact_{param}.png', dpi=300, bbox_inches='tight')
                        plt.close()
                    except Exception as e:
                        print(f"Error creating parameter impact plot for {param}: {e}")
                
                # 4. Heatmap for pairs of parameters (if applicable)
                if len(varying_params) >= 2:
                    for param1, param2 in itertools.combinations(varying_params[:3], 2):  # Limit to top 3 parameters
                        try:
                            plt.figure(figsize=(10, 8))
                            
                            # Convert parameters to string
                            df[f"{param1}_str"] = df[param1].astype(str)
                            df[f"{param2}_str"] = df[param2].astype(str)
                            
                            # Create pivot table
                            pivot = df.pivot_table(
                                values='f1_score', 
                                index=f"{param1}_str", 
                                columns=f"{param2}_str", 
                                aggfunc='mean'
                            )
                            
                            # Plot heatmap
                            sns.heatmap(pivot, annot=True, cmap='viridis', fmt='.3f')
                            plt.title(f'F1 Score by {param1} and {param2}')
                            plt.tight_layout()
                            plt.savefig(figures_dir / f'heatmap_{param1}_{param2}.png', dpi=300, bbox_inches='tight')
                            plt.close()
                        except Exception as e:
                            print(f"Error creating heatmap for {param1} and {param2}: {e}")
        # 4. Plot overfitting gap by experiment
        if 'overfitting_gap' in df.columns and not df['overfitting_gap'].isnull().all():
            plt.figure(figsize=(12, 6))
            plt.bar(df_sorted[x_axis_col], df_sorted['overfitting_gap'])
            plt.xlabel('Experiment')
            plt.ylabel('Overfitting Gap (Train F1 - Test F1)')
            plt.title('Overfitting Gap Across Experiments')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_path = figures_dir / 'overfitting_gaps.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved overfitting analysis to {save_path}")
        
        
    except Exception as e:
        print(f"Error in plot_experiment_results: {e}")
        traceback.print_exc()

# # Placeholder for save_figure if it's defined elsewhere, otherwise define simple one
def save_figure(fig, name, directory):
    try:
        path = Path(directory) / f"{name}.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Saved figure: {path}")
    except Exception as e:
        print(f"Error saving figure {name}: {e}")

def plot_experiment_results(summary_df, batch_dir):
    """
    Create visualizations of experiment results using Matplotlib.

    Args:
        summary_df: DataFrame with experiment results
        batch_dir: Batch directory path
    """
    print("\n--- Generating Experiment Result Visualizations ---")
    if not isinstance(summary_df, pd.DataFrame) or summary_df.empty:
         print("Summary DataFrame is empty or invalid. Skipping visualization.")
         return

    # Check if there are successful experiments
    if 'f1_score' not in summary_df.columns or summary_df['f1_score'].isnull().all():
        print("No successful experiments with F1 scores to visualize.")
        return

    # Drop rows with errors (no f1_score) for plotting main results
    df = summary_df.dropna(subset=['f1_score']).copy() # Use copy to avoid SettingWithCopyWarning

    # Check if we have any valid data to plot
    if len(df) == 0:
        print("No experiments with valid F1 scores to visualize after dropping NaNs.")
        return

    # Create figures directory
    figures_dir = Path(batch_dir) / 'figures'
    figures_dir.mkdir(exist_ok=True)
    print(f"Saving plots to: {figures_dir}")

    try:
        # --- Determine X-axis Label ---
        # Prefer 'name' column if it exists and is unique enough, otherwise use 'experiment_id'
        if 'name' in df.columns and df['name'].nunique() == len(df):
            x_axis_col = 'name'
            print("Using 'name' for x-axis labels.")
        elif 'experiment_id' in df.columns:
             x_axis_col = 'experiment_id'
             print("Using 'experiment_id' for x-axis labels.")
        else:
            # If neither column exists, create a simple index as fallback
            print("Warning: Neither 'name' nor 'experiment_id' found. Using generated plot index for labels.")
            df['plot_index'] = [f"Exp_{i+1}" for i in range(len(df))]
            x_axis_col = 'plot_index'

        # Sort by F1 score for the main comparison plot
        df_sorted = df.sort_values('f1_score', ascending=False).reset_index(drop=True)

        # --- Plot 1: Bar chart of F1 scores (using Matplotlib) ---
        print("Generating F1 score comparison plot...")
        plt.figure(figsize=(max(10, len(df_sorted) * 0.6), 7)) # Dynamic width, fixed height
        plot_labels_f1 = df_sorted[x_axis_col].astype(str)
        x_pos_f1 = np.arange(len(df_sorted))

        bars = plt.bar(x_pos_f1, df_sorted['f1_score'], color='skyblue', label='F1 Score')

        # Add F1 score labels on top
        for bar in bars:
             yval = bar.get_height()
             plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom', fontsize=8, rotation=90)

        plt.xlabel('Experiment')
        plt.ylabel('F1 Score')
        plt.title('F1 Scores Across Experiments (Sorted)')
        plt.xticks(x_pos_f1, plot_labels_f1, rotation=70, ha='right', fontsize=9) # Increased rotation
        plt.ylim(bottom=max(0, df_sorted['f1_score'].min() - 0.05), top=min(1.05, df_sorted['f1_score'].max() + 0.05)) # Dynamic ylim
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout(pad=1.5) # Add padding
        save_figure(plt.gcf(), "f1_score_comparison", figures_dir)
        plt.close()

        # --- Plot 2: Execution times (if available) ---
        if 'execution_time' in df.columns and not df['execution_time'].isnull().all():
             print("Generating execution time plot...")
             # Sort by execution time for this plot
             df_time_sorted = df.sort_values('execution_time', ascending=False).reset_index(drop=True)
             plot_labels_time = df_time_sorted[x_axis_col].astype(str)
             x_pos_time = np.arange(len(df_time_sorted))

             plt.figure(figsize=(max(10, len(df_time_sorted) * 0.6), 7))
             plt.bar(x_pos_time, df_time_sorted['execution_time'] / 60, color='lightcoral') # Convert seconds to minutes
             plt.xlabel('Experiment')
             plt.ylabel('Execution Time (minutes)')
             plt.title('Execution Times Across Experiments (Sorted)')
             plt.xticks(x_pos_time, plot_labels_time, rotation=70, ha='right', fontsize=9)
             plt.grid(axis='y', linestyle='--', alpha=0.7)
             plt.tight_layout(pad=1.5)
             save_figure(plt.gcf(), "execution_times", figures_dir)
             plt.close()

        # --- Plot 3: Parameter influence analysis ---
        print("Analyzing parameter influence...")
        # Identify parameters that actually vary across the successful runs
        param_columns = [col for col in df.columns
                         if col not in ['experiment_id', 'name', 'plot_index', # Exclude identifiers
                                        'best_model_name', 'report_dir', # Exclude run outputs
                                        'accuracy', 'f1_score', 'execution_time', # Exclude main metrics
                                        'status', 'error_message', 'overfitting_gap', # Exclude status/derived metrics
                                        'train_f1', 'test_f1']] # Exclude other metrics

        varying_params = []
        for param in param_columns:
            try:
                 # Check number of unique non-null values
                 # Convert potential unhashable types to strings first for uniqueness check
                 unique_vals = df[param].dropna().apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, dict) else json.dumps(x) if isinstance(x, list) else x).nunique()
                 if unique_vals > 1:
                     varying_params.append(param)
            except TypeError as e:
                 print(f"  Skipping parameter '{param}' for influence analysis due to unhashable type issue during check: {e}")
            except Exception as e_gen:
                 print(f"  Error checking uniqueness for parameter '{param}': {e_gen}")


        print(f"  Found varying parameters: {varying_params}")

        if varying_params:
            # Plot the influence of each varying parameter on F1 score
            for param in varying_params:
                print(f"  Generating plot for F1 vs '{param}'...")
                try:
                    plt.figure(figsize=(10, 7))

                    grouped_stats = defaultdict(list)
                # fix for unhashable list param like feat idx: Get unique string representations of parameter values
                    if param in df.columns:
                        unique_param_values = df[param].dropna().apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, dict) else json.dumps(x) if isinstance(x, list) else str(x)).unique()
                    else:
                        print(f" Column '{param}' not found in DataFrame. Skipping plot.")
                        plt.close()
                        continue
               
                    
                    # old
                    # unique_param_values = df[param].dropna().unique() # Get unique actual values
                    for param_value in unique_param_values:
                        # Convert dict/list/etc. to a string key for grouping
                        # if isinstance(param_value, dict):
                        #      group_key_str = json.dumps(param_value, sort_keys=True)
                        # elif isinstance(param_value, list):
                        #      group_key_str = json.dumps(param_value)
                        # else:
                        #      group_key_str = str(param_value) # Convert others to string too

                        # Filter DataFrame based on original parameter value (handle NaNs comparison)
                        if pd.isna(param_value):
                             f1_values_for_group = df[df[param].isnull()]['f1_score']
                        else:
                             # Apply same conversion logic to the column for comparison
                            #  f1_values_for_group = df[df[param].apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, dict) else json.dumps(x) if isinstance(x, list) else str(x)) == group_key_str]['f1_score']
                            f1_values_for_group = df[df[param].apply(lambda x: json.dumps(x, sort_keys=True) if isinstance(x, dict) else json.dumps(x) if isinstance(x, list) else str(x)) == param_value]['f1_score']

                        if not f1_values_for_group.empty:
                            #  grouped_stats[group_key_str].extend(f1_values_for_group.tolist())
                             grouped_stats[param_value].extend(f1_values_for_group.tolist())

                # --- END: Fix for unhashable types ---

                    if not grouped_stats:
                         print(f"    No data found for parameter '{param}'. Skipping plot.")
                         plt.close()
                         continue

                    # Calculate mean and std dev for plotting error bars
                    group_labels = []
                    group_means = []
                    group_stds = []
                    sorted_keys = sorted(grouped_stats.keys()) # Sort by string key

                    for key in sorted_keys:
                        values = grouped_stats[key]
                        if not values: continue # Skip empty groups
                        # Create label (shorten if necessary)
                        label = key
                        if len(label) > 40: label = label[:37] + "..."
                        group_labels.append(label)
                        group_means.append(np.mean(values))
                        group_stds.append(np.std(values))

                    # Plotting
                    x_pos_group = np.arange(len(group_labels))
                    plt.errorbar(x_pos_group, group_means, yerr=group_stds, fmt='o', color='darkcyan', ecolor='lightblue', elinewidth=3, capsize=5, label='Mean F1 ± Std Dev')

                    # Optional: Plot individual points slightly jittered
                    all_x = []
                    all_y = []
                    for i, key in enumerate(sorted_keys):
                        values = grouped_stats[key]
                        jitter = np.random.normal(0, 0.05, size=len(values))
                        all_x.extend([i + j for j in jitter])
                        all_y.extend(values)
                    plt.scatter(all_x, all_y, alpha=0.4, s=20, label='Individual Runs', color='orange')

                    plt.xticks(x_pos_group, group_labels, rotation=45, ha='right', fontsize=9)
                    plt.xlabel(param.replace('_', ' ').title())
                    plt.ylabel('F1 Score')
                    plt.title(f'F1 Score vs {param.replace("_", " ").title()}')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.6)
                    plt.tight_layout()
                    save_figure(plt.gcf(), f"param_impact_{param}", figures_dir)
                    plt.close()

                except Exception as e:
                    print(f"  Error creating parameter impact plot for '{param}': {e}")
                    traceback.print_exc()
                    plt.close() # Close figure on error

        # --- Plot 4: Overfitting gap (if available) ---
        if 'overfitting_gap' in df.columns and not df['overfitting_gap'].isnull().all():
             print("Generating overfitting gap plot...")
             # Sort by gap for this plot
             df_gap_sorted = df.sort_values('overfitting_gap', ascending=False).reset_index(drop=True)
             # Use the same x-axis labels as the F1 plot for consistency, but reordered
             plot_labels_gap = df_gap_sorted[x_axis_col].astype(str)
             x_pos_gap = np.arange(len(df_gap_sorted))

             plt.figure(figsize=(max(10, len(df_gap_sorted) * 0.6), 7))
             plt.bar(x_pos_gap, df_gap_sorted['overfitting_gap'], color='salmon')
             plt.xlabel('Experiment')
             plt.ylabel('Overfitting Gap (Train F1 - Test F1)')
             plt.title('Overfitting Gap Across Experiments (Sorted by Gap)')
             plt.xticks(x_pos_gap, plot_labels_gap, rotation=70, ha='right', fontsize=9)
             plt.grid(axis='y', linestyle='--', alpha=0.7)
             # Add a horizontal line threshold for warning levels
             plt.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, label='Moderate Overfitting Threshold (0.1)')
             plt.axhline(y=0.2, color='red', linestyle='--', linewidth=1, label='Severe Overfitting Threshold (0.2)')
             plt.legend(fontsize='small')
             plt.tight_layout(pad=1.5)
             save_figure(plt.gcf(), "overfitting_gaps", figures_dir)
             plt.close()

        print("--- Visualization Generation Complete ---")

    except Exception as e:
        # Use globally imported traceback
        print(f"Error during plot_experiment_results: {e}")
        traceback.print_exc()
        plt.close('all') # Close any potentially open figures on error


def create_combined_report(best_models, best_f1_scores, batch_dir):
    """
    Create a combined report of the best models
    
    Args:
        best_models: Dictionary of best models
        best_f1_scores: Dictionary of best F1 scores
        batch_dir: Batch directory path
    """
    # Create a summary of the best models
    summary = []
    
    try:
        # Filter out entries with None values
        valid_models = {}
        valid_scores = {}
        
        for config_key in best_models:
            if config_key in best_f1_scores and best_f1_scores[config_key] is not None:
                valid_models[config_key] = best_models[config_key]
                valid_scores[config_key] = best_f1_scores[config_key]
        
        if not valid_models:
            print("No valid models to include in combined report")
            return
            
        for config_key in valid_models:
            model_info = valid_models[config_key]
            f1_score = valid_scores[config_key]
            
            summary.append({
                'experiment': config_key,
                'model_name': model_info['model_name'],
                'f1_score': f1_score,
                'report_dir': model_info['report_dir']
            })
        
        # Create DataFrame and sort by F1 score
        if not summary:
            print("No models to include in summary")
            return
            
        summary_df = pd.DataFrame(summary)
        summary_df = summary_df.sort_values('f1_score', ascending=False)
        
        # Save summary
        summary_df.to_csv(batch_dir / 'best_models_summary.csv', index=False)
        
        # Generate HTML report with links to individual experiment reports
        from jinja2 import Template
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Batch Experiment Summary</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #333; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
                .top-model { background-color: #e6ffe6; }
            </style>
        </head>
        <body>
            <h1>Batch Experiment Summary</h1>
            
            <h2>Best Models (Ranked by F1 Score)</h2>
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Experiment</th>
                    <th>Model</th>
                    <th>F1 Score</th>
                    <th>Report</th>
                </tr>
                {% for i, model in enumerate(models) %}
                <tr {% if i == 0 %}class="top-model"{% endif %}>
                    <td>{{ i+1 }}</td>
                    <td>{{ model.experiment }}</td>
                    <td>{{ model.model_name }}</td>
                    <td>{{ "%.4f"|format(model.f1_score) }}</td>
                    <td><a href="{{ model.report_dir }}/report.html">View Report</a></td>
                </tr>
                {% endfor %}
            </table>
            
            <h2>Experiment Visualizations</h2>
            <div>
                <h3>F1 Scores Across Experiments</h3>
                <img src="figures/f1_scores.png" alt="F1 Scores" style="max-width: 100%;">
            </div>
            
            <div>
                <h3>Execution Times</h3>
                <img src="figures/execution_times.png" alt="Execution Times" style="max-width: 100%;">
            </div>
        </body>
        </html>
        """
        
        # Render HTML
        template = Template(html_template)
        html_content = template.render(models=summary_df.to_dict(orient='records'), enumerate=enumerate)
        
        # Save HTML report
        html_path = batch_dir / 'batch_summary.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"Generated batch summary HTML report at {html_path}")
        
    except Exception as e:
        print(f"Error creating combined report: {e}")
        traceback.print_exc()
        
def generate_experiment_configurations(param_grid):
    """
    Generate experiment configurations from a parameter grid
    
    Args:
        param_grid: Parameter grid dictionary
    
    Returns:
        List of configuration dictionaries
    """
    # Use ParameterGrid for efficient generation
    grid = ParameterGrid(param_grid)
    configs = list(grid)
    
    print(f"Generated {len(configs)} experiment configurations")
    return configs

def run_batch_experiments(data_path, param_grid=None, configs=None, base_report_dir="reports/batch_experiments",
                         max_experiments=None, save_best_only=False, save_reports=True):
    """
    Run a batch of experiments with different configurations
    
    Args:
        data_path: Path to data
        param_grid: Parameter grid for generating configurations
        configs: List of predefined configurations (alternative to param_grid)
        base_report_dir: Base directory for reports
        max_experiments: Maximum number of experiments to run
        save_best_only: Whether to save only the best model in each experiment
        save_reports: Whether to save detailed reports for each experiment
        
    Returns:
        best_overall_model: Best overall model
        results_summary: Summary of all experiment results
    """
    
    # Ensure either param_grid or configs is provided
    if param_grid is None and configs is None:
        raise ValueError("Either param_grid or configs must be provided")
    
    # Generate configurations if param_grid is provided
    if configs is None:
        configs = generate_experiment_configurations(param_grid)
    
    # Limit number of experiments if specified
    if max_experiments is not None and max_experiments < len(configs):
        print(f"Limiting to {max_experiments} experiments out of {len(configs)} configurations")
        configs = configs[:max_experiments]
    
    # Create base report directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = Path(base_report_dir) / f"batch_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Save all configurations
    with open(batch_dir / 'all_configurations.json', 'w') as f:
        json.dump(configs, f, indent=4)
    
    # Initialize results storage
    results_summary = []
    best_models = {}
    best_f1_scores = {}
    
    # Run experiments
    for i, config in enumerate(configs):
        experiment_name = f"experiment_{i+1}"
        print(f"\n{'='*80}")
        print(f"Experiment {i+1}/{len(configs)}")
        print(f"Configuration: {config}")
        print(f"{'='*80}")
        
        # Start timer
        start_time = time.time()
        
        # Create report directory for this experiment
        if save_reports:
            report_dir, subdirs = create_report_directory(base_dir=str(batch_dir), experiment_name=experiment_name)
            # Save experiment configuration
            save_experiment_config(config, report_dir)
        else:
            # Create a minimal directory structure for saving models
            report_dir = batch_dir / experiment_name
            report_dir.mkdir(exist_ok=True)
            subdirs = {'models': report_dir / 'models'}
            subdirs['models'].mkdir(exist_ok=True)
        
        try:
            # Extract configuration parameters
            model_params = config.copy()
            
            # Add data_path to parameters
            model_params['data_path'] = data_path
            
            # Add report_dir to parameters
            model_params['report_dir'] = str(report_dir)
            
            
            # Run classification pipeline with report directory
            best_model, experiment_results = run_classification_pipeline_with_reporting(**model_params)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Extract key results
            best_model_name = experiment_results.get('best_model_name', 'unknown')
            accuracy = experiment_results.get('accuracy', 0)
            f1_score = experiment_results.get('f1_score', 0)
            
            # Store in summary
            summary_entry = {
                'experiment_id': i+1,
                'experiment_name': experiment_name,
                'best_model_name': best_model_name,
                'accuracy': accuracy,
                'f1_score': f1_score,
                'execution_time': execution_time,
                'report_dir': str(report_dir),
                'train_f1': experiment_results.get('train_f1', 0),  # Add training metrics
                'overfitting_gap': experiment_results.get('overfitting_gap', 0),  # Add overfitting gap
                **config  # Include all configuration parameters
            }
            results_summary.append(summary_entry)
            
            # Save information about the best model
            config_key = f"experiment_{i+1}"
            best_models[config_key] = {
                'model': best_model,
                'model_name': best_model_name,
                'report_dir': str(report_dir)
            }
            best_f1_scores[config_key] = f1_score
            
            print(f"\nBest model for experiment {i+1}: {best_model_name}")
            print(f"F1 Score: {f1_score:.4f}")
            # print(f"Training F1: {summary_entry['train_f1']:.4f} (Gap: {summary_entry['overfitting_gap']:.4f})") # added
            print(f"Execution time: {execution_time:.2f} seconds")
  
            
        except Exception as e:
            print(f"Error in experiment {i+1}: {str(e)}")
            # Store error in summary
            summary_entry = {
                'experiment_id': i+1,
                'experiment_name': experiment_name,
                'status': 'error',
                'error_message': str(e),
                'report_dir': str(report_dir),
                **config
            }
            results_summary.append(summary_entry)
            traceback.print_exc()
    
    # Save summary report
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(batch_dir / 'experiment_summary.csv', index=False)
    
    # Find best overall model - WITH FIX FOR NoneType COMPARISON ERROR
    best_overall_model = None
    best_overall_f1 = -1  # Use -1 as a sentinel value
    
    if best_f1_scores:
        try:
            # Filter out any None values or non-numeric values
            valid_scores = {k: v for k, v in best_f1_scores.items() 
                          if v is not None and isinstance(v, (int, float))}
            
            if valid_scores:  # Only proceed if we have valid scores
                best_overall_config = max(valid_scores, key=valid_scores.get)
                best_overall_model_info = best_models[best_overall_config]
                best_overall_model = best_overall_model_info['model']
                best_overall_model_name = best_overall_model_info['model_name']
                best_overall_f1 = valid_scores[best_overall_config]
                best_overall_report_dir = best_overall_model_info['report_dir']
                
                print(f"\n{'='*80}")
                print(f"Best overall experiment: {best_overall_config}")
                print(f"Best overall model: {best_overall_model_name}")
                print(f"Best overall F1 Score: {best_overall_f1:.4f}")
                print(f"Report directory: {best_overall_report_dir}")
                print(f"{'='*80}")
                
                # Create a visualization of all experiment results
                plot_experiment_results(summary_df, batch_dir)
                
                # Create a combined report of the best models
                create_combined_report(best_models, best_f1_scores, batch_dir)
            else:
                print("No valid F1 scores found in experiments.")
        except Exception as e:
            print(f"Error finding best model: {e}")
            traceback.print_exc()
            
    if best_overall_model is None:
        print("No successful experiments found")
    
    return best_overall_model, pd.DataFrame(results_summary)