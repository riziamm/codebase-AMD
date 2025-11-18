
import sys
import os
import json
import argparse
from pathlib import Path
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from collections import Counter 
import traceback 
import matplotlib
# Force matplotlib to not use any Xwindows backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import pip
pip.main(['install','seaborn'])
import seaborn as sns

def save_experiment_config(config, report_dir):
    """
    Save experiment configuration to a JSON file
    
    Args:
        config: Experiment configuration dictionary
        report_dir: Report directory path
    """
    # Create a serializable copy of the config
    serializable_config = config.copy()
    
    # Convert any non-serializable values
    for key, value in serializable_config.items():
        if isinstance(value, np.ndarray):
            serializable_config[key] = value.tolist()
        elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            serializable_config[key] = str(value)
    
    # Save to JSON file
    config_path = Path(report_dir) / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=4)
    
    print(f"Saved experiment configuration to {config_path}")

def save_model_metrics(results, report_dir, model_name):
    """
    Save model metrics to CSV and JSON files
    
    Args:
        results: Dictionary of model results
        report_dir: Report directory path
        model_name: Name of the model
    """
    metrics_dir = Path(report_dir) / 'metrics'
    
    # Extract metrics
    metrics = {
        'model_name': model_name,
        'accuracy': results.get('accuracy', 0),
        'f1_score': results.get('f1_score', 0),
        # 'precision': 0,
        # 'recall': 0,
        'precision': results.get('precision', 0),
        'recall': results.get('recall', 0),
        'roc_auc': results.get('roc_auc_scores',0.0),
        'avg_precision':results.get('avg_precision_scores', 0.0),
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Extract precision and recall from classification_report if available
    if 'classification_report' in results:
        # Get weighted avg metrics which includes precision and recall
        if 'weighted avg' in results['classification_report']:
            metrics['precision'] = results['classification_report']['weighted avg']['precision']
            metrics['recall'] = results['classification_report']['weighted avg']['recall']
    
    # Save as CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_dir / f"{model_name}_metrics.csv", index=False)
    
    # Save as JSON
    with open(metrics_dir / f"{model_name}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # If detailed metrics are available (like classification report)
    if 'classification_report' in results:
        with open(metrics_dir / f"{model_name}_classification_report.json", 'w') as f:
            json.dump(results['classification_report'], f, indent=4)
    
    print(f"Saved metrics for {model_name}")

def save_figure(fig, filename, report_dir, dpi=300):
    """
    Save a matplotlib figure to the figures directory
    
    Args:
        fig: Matplotlib figure object
        filename: Name of the file (without extension)
        report_dir: Report directory path
        dpi: DPI for the saved figure
    """
    figures_dir = Path(report_dir) / 'figures'
    filepath = figures_dir / f"{filename}.png"
    
    # Save the figure
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure to {filepath}")

def create_visualization_plots(model, X_train, X_test, y_train, y_test, feature_names, report_dir, model_name, le=None):
    """
    Create and save visualization plots
    
    Args:
        model: Trained model
        X_train, X_test, y_train, y_test: Training and test data
        feature_names: Names of features
        report_dir: Report directory path
        model_name: Name of the model
        le: Label encoder (optional)
    """
    figures_dir = Path(report_dir) / 'figures'
    
    # 1. Feature Importance (if available)
    try:
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            importances = model.feature_importances_
            indices = np.argsort(importances)[-20:]  # Top 20 features
            
            plt.barh(range(len(indices)), importances[indices])
            plt.yticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f"Feature {i}" for i in indices])
            plt.xlabel('Feature Importance')
            plt.title(f'Top Feature Importances - {model_name}')
            save_figure(plt.gcf(), f"{model_name}_shap_feature_importance", report_dir)
            plt.close()
    except Exception as e:
        print(f"Warning: Could not generate feature importance plot. Error: {e}")
    

    # 2. Confusion Matrix
    try:
        y_pred = model.predict(X_test)
        
        plt.figure(figsize=(8, 6))
        
        # For binary classification with label encoder
        if len(np.unique(y_test)) == 2 and le is not None:
            # Get true binary labels
            binary_labels = np.unique(y_test)
            
            # Handle case where we're doing binary but with the original multiclass data
            if hasattr(le, 'classes_') and len(le.classes_) > 2:
                # Check if we're doing one vs rest classification
                unique_pred = np.unique(y_pred)
                unique_test = np.unique(y_test)
                
                if len(unique_pred) <= 2 and len(unique_test) <= 2:
                    # Create explicit labels for binary case
                    label_map = {0: "Class 0", 1: "Other Classes"}
                    
                    # Create confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Create heatmap with explicit binary labels
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=list(label_map.values()), 
                            yticklabels=list(label_map.values()))
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title(f'Confusion Matrix - {model_name}')
                else:
                    # Handle as multiclass
                    # Use original class names from label encoder
                    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                            xticklabels=le.classes_, yticklabels=le.classes_)
                    plt.xlabel('Predicted')
                    plt.ylabel('True')
                    plt.title(f'Confusion Matrix - {model_name}')
            else:
                # Binary classification with binary labels
                if hasattr(le, 'classes_'):
                    labels = le.classes_
                else:
                    labels = [f"Class {i}" for i in binary_labels]
                    
                # Create confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                
                # Create heatmap with binary labels
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=labels, yticklabels=labels)
                plt.xlabel('Predicted')
                plt.ylabel('True')
                plt.title(f'Confusion Matrix - {model_name}')
        else:
            # Multiclass case
            if le is not None and hasattr(le, 'classes_'):
                # Use original class names
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                        xticklabels=le.classes_, yticklabels=le.classes_)
            else:
                # Use generic class names
                class_labels = [f"Class {i}" for i in range(len(np.unique(y_test)))]
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                        xticklabels=class_labels, yticklabels=class_labels)
            
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {model_name}')
        
        # Save the figure
        save_figure(plt.gcf(), f"{model_name}_anls_conf_matrix", report_dir)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate confusion matrix plot. Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. ROC Curve (for binary classification)
    try:
        if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
            plt.figure(figsize=(8, 6))
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            
            plt.plot(fpr, tpr, label=f'ROC curve')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {model_name}')
            plt.legend()
            save_figure(plt.gcf(), f"{model_name}_anls_roc_curve", report_dir)
            plt.close()
            
            # 4. Precision-Recall Curve
            plt.figure(figsize=(8, 6))
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {model_name}')
            save_figure(plt.gcf(), f"{model_name}_anls_precision_recall_curve", report_dir)
            plt.close()
    except Exception as e:
        print(f"Warning: Could not generate ROC or PR curve. Error: {e}")
    
    # 5. Learning Curve (simplified version)
    # Within create_visualization_plots function, in the Learning Curve section:
    try:
        from sklearn.model_selection import learning_curve
        
        plt.figure(figsize=(10, 6))
        train_sizes = np.linspace(0.1, 1.0, 10)
        
        # Check if cv is too large for the minority class
        class_counts = np.bincount(y_train)
        min_samples = min(class_counts[class_counts > 0])  # Ignore classes with zero samples
        
        # Use max 2 splits for very small classes (< 5 samples)
        if min_samples < 5:
            cv = 2
        # Use 3 splits for small classes (5-10 samples)
        elif min_samples < 10:
            cv = 3
        # Use default for larger classes
        else:
            cv = 5

        print(f"Using {cv}-fold cross-validation (smallest class has {min_samples} samples)")
        
        # # Adjust cv based on minority class size
        # cv = 5  # Default
        # if min_samples < cv:
        #     # Use at most min_samples splits, but at least 2 if possible
        #     cv = max(2, min(min_samples, 3))
        #     print(f"Adjusting cross-validation: Minority class has only {min_samples} samples, using {cv} folds")
        
        # # Limit n_jobs to avoid memory issues
        n_jobs = min(2, os.cpu_count() or 1)
        
        # Calculate learning curve
        train_sizes, train_scores, validation_scores = learning_curve(
            model, X_train, y_train, train_sizes=train_sizes, cv=cv, scoring='f1_weighted',
            n_jobs=n_jobs, shuffle=True, random_state=42
        )
    
        # Calculate mean and std for training and validation scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        validation_mean = np.mean(validation_scores, axis=1)
        validation_std = np.std(validation_scores, axis=1)
        
        # Plot the learning curve
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
        plt.fill_between(train_sizes, validation_mean - validation_std, validation_mean + validation_std, alpha=0.1, color="green")
        plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training score")
        plt.plot(train_sizes, validation_mean, 's-', color="green", label="Cross-validation score")
        
        # Plot test score if available
        from sklearn.metrics import f1_score
        test_score = f1_score(y_test, y_pred, average='weighted')
        plt.axhline(y=test_score, color='r', linestyle='-', label=f'Test score: {test_score:.4f}')
        
        plt.xlabel('Training Examples')
        plt.ylabel('F1 Score')
        plt.title(f'Learning Curve - {model_name}')
        plt.legend(loc="best")
        plt.grid()
        save_figure(plt.gcf(), f"{model_name}_anls_learning_curve", report_dir)
        plt.close()
    except Exception as e:
        print(f"Warning: Could not generate learning curve. Error: {e}")

        
        
def generate_html_report(report_dir, experiment_config, results):
    """
    Generate HTML report summarizing the experiment with improved organization
    
    Args:
        report_dir: Report directory path
        experiment_config: Experiment configuration
        results: Dictionary of results
    """
    
    from jinja2 import Template
    import base64
    import os
    
    try:
        # Get list of figures
        figures_dir = Path(report_dir) / 'figures'
        figure_files = list(figures_dir.glob('*.png'))
        
        # Debug: list all figure files to see what's available
        print(f"Found {len(figure_files)} figure files in {figures_dir}")
        for fig_path in figure_files:
            print(f"Found figure: {fig_path.name}")

        # Initialize collections before using them
        all_figures = {}
        overview_figures = []
        shap_figures = []
        model_figures = {}
        metrics = []
        model_reports = {}
        best_model_name = results.get('best_model_name', '')
        
        # First pass: collect all figure names and data
        for fig_path in figure_files:
            try:
                with open(fig_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                    
                fig_name = fig_path.stem
                all_figures[fig_name] = {
                    'name': fig_name,
                    'data': img_data,
                    'file_size': os.path.getsize(fig_path)
                }
                print(f"Loaded figure: {fig_name}")
            except Exception as e:
                print(f"Error loading figure {fig_path}: {e}")
        
        # # Extract list of model names from metrics or results
        # model_names = []
        # if 'all_models' in results and results['all_models']:
        #     model_names = list(results['all_models'].keys())
        # else:
        #     # Try to get model names from metrics directory
        #     metrics_dir = Path(report_dir) / 'metrics'
        #     metric_files = list(metrics_dir.glob('*_metrics.json'))
        #     for metric_file in metric_files:
        #         model_name = metric_file.stem.replace('_metrics', '')
        #         if model_name not in model_names:
        #             model_names.append(model_name)
        
                    
            # # Try to identify model names from figures
            # if not model_names:
            #     for fig_name in all_figures:
            #         parts = fig_name.split('_')
            #         if len(parts) > 1:
            #             potential_model = ' '.join(parts[:-1])
            #             if potential_model not in model_names:
            #                 model_names.append(potential_model)
            
        # Extract list of model names from metrics or results
        model_names = []
        if 'all_models' in results:
            model_names = list(results['all_models'].keys())
        else:
            # Try to get model names from metrics directory
            metrics_dir = Path(report_dir) / 'metrics'
            metric_files = list(metrics_dir.glob('*_metrics.json'))
            for metric_file in metric_files:
                model_name = metric_file.stem.replace('_metrics', '')
                if model_name not in model_names:
                    model_names.append(model_name)
        
        # Initialize model figures dictionary
        for model_name in model_names:
            model_figures[model_name] = {'plots': [], 'shap': []}
            
        # # First categorize overview figures
        # for fig_name, fig_data in all_figures.items():
        #     fig_name_lower = fig_name.lower()
            
        #     # Overview figures
        #     if ("class_distribution" in fig_name_lower or 
        #         "model_comparison" in fig_name_lower or 
        #         "confusion_matrix_comparison" in fig_name_lower):
        #         overview_figures.append(fig_data)
        #         print(f"Added to overview figures: {fig_name}")
        #         continue

        # # Next, categorize SHAP figures and assign to models
        # for fig_name, fig_data in all_figures.items():
        #     fig_name_lower = fig_name.lower()
            
        #     # Skip figures already categorized
        #     if fig_data in overview_figures:
        #         continue
            
        #     # Check if it's a SHAP/importance figure
        #     is_shap = any(term in fig_name_lower for term in [
        #         'shap', 'waterfall', 'force', 'beeswarm', 
        #         'feature_importance', 'importance', 'coefficient'
        #     ])
            
        #     if is_shap:
        #         # Try to find which model it belongs to
        #         model_found = False
        #         for model_name in model_names:
        #             model_variants = [
        #                 model_name.lower(),
        #                 model_name.replace(' ', '_').lower(),
        #                 model_name.replace(' ', '').lower(),
        #                 ''.join(word[0].lower() for word in model_name.split())  # acronym
        #             ]
                    
        #             if any(variant in fig_name_lower for variant in model_variants):
        #                 # Add to model's SHAP figures
        #                 model_figures[model_name]['shap'].append(fig_data)
        #                 print(f"Added to {model_name} SHAP figures: {fig_name}")
        #                 model_found = True
        #                 break
                
        #         # If no model was found, add to general SHAP figures
        #         if not model_found:
        #             shap_figures.append(fig_data)
        #             print(f"Added to general SHAP figures: {fig_name}")
                
        #         # Skip further processing for this figure
        #         continue

        # # Finally categorize other model-specific figures
        # for fig_name, fig_data in all_figures.items():
        #     fig_name_lower = fig_name.lower()
            
        #     # Skip figures already categorized
        #     if (fig_data in overview_figures or 
        #         fig_data in shap_figures or 
        #         any(fig_data in figs.get('shap', []) for figs in model_figures.values())):
        #         continue
            
        #     # Try to find which model it belongs to
        #     for model_name in model_names:
        #         model_variants = [
        #             model_name.lower(),
        #             model_name.replace(' ', '_').lower(),
        #             model_name.replace(' ', '').lower(),
        #             ''.join(word[0].lower() for word in model_name.split())  # acronym
        #         ]
                
        #         if any(variant in fig_name_lower for variant in model_variants):
        #             # Add to model's regular plots
        #             model_figures[model_name]['plots'].append(fig_data)
        #             print(f"Added to {model_name} regular plots: {fig_name}")
        #             break
        
        # More robust categorization
        for fig_name, fig_data in all_figures.items():
            fig_name_lower = fig_name.lower()
            
            # Overview figures
            if ("class_distribution" in fig_name_lower or 
                "model_comparison" in fig_name_lower or 
                "confusion_matrix_comparison" in fig_name_lower):
                overview_figures.append(fig_data)
                continue
            
            # Check each model name
            matched_model = None
            for model_name in model_names:
                # Try different forms of the model name
                model_variations = [
                    model_name.lower(),
                    model_name.replace(' ', '_').lower(),
                    model_name.replace(' ', '').lower()
                ]
                
                if any(var in fig_name_lower for var in model_variations):
                    matched_model = model_name
                    break
            
            if matched_model:
                # Figure belongs to a specific model
                if "shap" in fig_name_lower:
                    model_figures[matched_model]['shap'].append(fig_data)
                else:
                    model_figures[matched_model]['plots'].append(fig_data)
            else:
                # General SHAP figure not associated with a model
                if "shap" in fig_name_lower:
                    shap_figures.append(fig_data)

        
        # Load consolidated metrics
        all_metrics_file = Path(report_dir) / 'metrics' / 'all_model_metrics.json'
        if all_metrics_file.exists():
            try:
                with open(all_metrics_file, 'r') as f:
                    all_metrics = json.load(f)
                
                metrics = []
                model_reports = {}
                best_model_name = all_metrics.get('best_model_name', '')
                
                for model_name, model_results in all_metrics['models'].items():
                    metrics.append({
                        'model_name': model_name,
                        'accuracy': model_results.get('accuracy', 0),
                        'f1_score': model_results.get('f1_score', 0),
                        'precision': model_results.get('precision', 0),
                        'recall': model_results.get('recall', 0),
                        'roc_auc':  model_results.get('roc_auc_scores',0.0),
                        'avg_precision': model_results.get('avg_precision_scores', 0.0)
                    })
                    
                    if 'classification_report' in model_results:
                        model_reports[model_name] = model_results['classification_report']
            except Exception as e:
                print(f"Error loading metrics from {all_metrics_file}: {e}")
                metrics = []
                model_reports = {}
                best_model_name = results.get('best_model_name', '')
        else:
            print(f"Warning: Consolidated metrics file not found at {all_metrics_file}")
            # Fall back to loading individual metrics files
            metrics_dir = Path(report_dir) / 'metrics'
            metrics_files = list(metrics_dir.glob('*_metrics.json'))
            
            metrics = []
            model_reports = {}
            for metrics_file in metrics_files:
                model_name = metrics_file.stem.replace('_metrics', '')
                with open(metrics_file, 'r') as f:
                    metrics.append(json.load(f))
                
                # Check if there's a classification report for this model
                report_file = metrics_dir / f"{model_name}_classification_report.json"
                if report_file.exists():
                    with open(report_file, 'r') as f:
                        model_reports[model_name] = json.load(f)
            
            best_model_name = results.get('best_model_name', '')
        
        # Sort metrics by F1 score (descending)
        metrics.sort(key=lambda x: x.get('f1_score', 0), reverse=True)
        
        # Debug information for troubleshooting
        print(f"Report generation state:")
        print(f"- Found {len(all_figures)} figures")
        print(f"- Identified {len(overview_figures)} overview figures")
        print(f"- Identified {len(shap_figures)} SHAP figures")
        print(f"- Categorized figures for {len(model_figures)} models")
        for model_name, figs in model_figures.items():
            print(f"  - {model_name}: {len(figs['plots'])} plots, {len(figs['shap'])} SHAP")
        print(f"- Collected metrics for {len(metrics)} models")
        print(f"- Best model name: {best_model_name}")
        
        # Debug check for SHAP figures
        print("\nSHAP Figure Debug Check:")
        print(f"General SHAP figures: {len(shap_figures)}")
        if len(shap_figures) > 0:
            for i, fig in enumerate(shap_figures):
                print(f"  {i+1}. {fig.get('name', 'Unnamed')}")

        for model_name, figs in model_figures.items():
            if 'shap' in figs and len(figs['shap']) > 0:
                print(f"SHAP figures for {model_name}: {len(figs['shap'])}")
                for i, fig in enumerate(figs['shap']):
                    print(f"  {i+1}. {fig.get('name', 'Unnamed')}")
        
        # Load HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Experiment Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                h1, h2, h3, h4 { color: #333; margin-top: 20px; }
                .section { margin-bottom: 40px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background-color: #fcfcfc; }
                .subsection { margin-bottom: 30px; }
                .figure-container { display: flex; flex-wrap: wrap; justify-content: center; gap: 20px; margin: 20px 0; }
                .figure { text-align: center; box-shadow: 0 0 5px rgba(0,0,0,0.1); padding: 10px; border-radius: 5px; background-color: white; }
                .figure img { max-width: 100%; max-height: 500px; border-radius: 3px; }
                .figure-caption { font-size: 14px; color: #555; margin-top: 8px; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; box-shadow: 0 0 5px rgba(0,0,0,0.1); }
                th, td { padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; font-weight: bold; }
                tr:hover { background-color: #f5f5f5; }
                .config { font-family: monospace; white-space: pre-wrap; padding: 15px; background-color: #f8f8f8; border-radius: 5px; overflow-x: auto; }
                .model-report { font-family: monospace; white-space: pre-wrap; padding: 15px; background-color: #f8f8f8; border-radius: 5px; margin: 15px 0; overflow-x: auto; }
                .best-model { background-color: rgba(76, 175, 80, 0.1); }
                .model-card { margin-bottom: 40px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; box-shadow: 0 0 10px rgba(0,0,0,0.05); background-color: white; }
                .best-model-card { border: 2px solid #4CAF50; }
                .summary-box { background-color: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; border-left: 5px solid #4CAF50; }
                .nav { position: sticky; top: 0; background-color: white; padding: 15px 0; z-index: 100; border-bottom: 1px solid #ddd; display: flex; gap: 10px; flex-wrap: wrap; }
                .nav-link { text-decoration: none; color: #333; padding: 8px 12px; border-radius: 5px; font-weight: bold; }
                .nav-link:hover { background-color: #f2f2f2; }
                .model-metrics { display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px; }
                .metric-box { flex: 1; min-width: 150px; padding: 15px; background-color: #f9f9f9; border-radius: 5px; text-align: center; }
                .metric-value { font-size: 24px; font-weight: bold; color: #333; margin: 5px 0; }
                .metric-label { font-size: 14px; color: #666; }
                .shap-section { margin-top: 30px; padding-top: 20px; border-top: 1px dashed #ddd; }
                .model-name { color: #2c3e50; font-weight: bold; }
                @media (max-width: 768px) {
                    .figure-container { flex-direction: column; }
                    .figure { max-width: 100%; }
                }
            </style>
        </head>
        <body>
            <h1>Machine Learning Experiment Report</h1>
            
            <div class="nav">
                <a href="#overview" class="nav-link">Overview</a>
                <a href="#data-analysis" class="nav-link">Data Analysis</a>
                <a href="#performance-summary" class="nav-link">Performance Summary</a>
                <a href="#model-details" class="nav-link">Model Details</a>
                {% if shap_figures or has_shap_plots %}
                <a href="#shap-analysis" class="nav-link">SHAP Analysis</a>
                {% endif %}
            </div>
            
            <div id="overview" class="section">
                <h2>Experiment Overview</h2>
                
                <div class="summary-box">
                    <h3>Best Model: <span class="model-name">{{ best_model_name }}</span></h3>
                    {% for metric in metrics %}
                        {% if metric.model_name == best_model_name %}
                        <div class="model-metrics">
                            <div class="metric-box">
                                <div class="metric-label">Accuracy</div>
                                <div class="metric-value">{{ "%.4f"|format(metric.accuracy) }}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">F1 Score</div>
                                <div class="metric-value">{{ "%.4f"|format(metric.f1_score) }}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">Precision</div>
                                <div class="metric-value">{{ "%.4f"|format(metric.precision) }}</div>
                            </div>
                            <div class="metric-box">
                                <div class="metric-label">Recall</div>
                                <div class="metric-value">{{ "%.4f"|format(metric.recall) }}</div>
                            </div>
                        </div>
                        {% endif %}
                    {% endfor %}
                </div>
                
                <h3>Experiment Configuration</h3>
                <pre class="config">{{ config }}</pre>
            </div>
            
            <div id="data-analysis" class="section">
                <h2>Data Analysis</h2>
                
                <div class="figure-container">
                    {% for figure in overview_figures %}
                    <div class="figure">
                        <img src="data:image/png;base64,{{ figure.data }}" alt="{{ figure.name }}">
                        <p class="figure-caption">{{ figure.name }}</p>
                    </div>
                    {% endfor %}
                </div>
            </div>
            
            <div id="performance-summary" class="section">
                <h2>Performance Summary</h2>
                
                <div class="subsection">
                    <h3>Model Metrics Comparison</h3>
                    <table>
                        <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>F1 Score</th>
                            <th>Precision</th>
                            <th>Recall</th>
                        </tr>
                        {% for metric in metrics %}
                        <tr {% if metric.model_name == best_model_name %}class="best-model"{% endif %}>
                            <td><a href="#model-{{ metric.model_name|replace(' ', '-')|lower }}">{{ metric.model_name }}</a></td>
                            <td>{{ "%.4f"|format(metric.accuracy) }}</td>
                            <td>{{ "%.4f"|format(metric.f1_score) }}</td>
                            <td>{{ "%.4f"|format(metric.precision) }}</td>
                            <td>{{ "%.4f"|format(metric.recall) }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
            </div>
            
            <div id="model-details" class="section">
                <h2>Model Details</h2>
                
                {% for metric in metrics %}
                <div class="model-card {% if metric.model_name == best_model_name %}best-model-card{% endif %}" id="model-{{ metric.model_name|replace(' ', '-')|lower }}">
                    <h3>{{ metric.model_name }} {% if metric.model_name == best_model_name %}(Best Model){% endif %}</h3>
                    
                    <div class="model-metrics">
                        <div class="metric-box">
                            <div class="metric-label">Accuracy</div>
                            <div class="metric-value">{{ "%.4f"|format(metric.accuracy) }}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">F1 Score</div>
                            <div class="metric-value">{{ "%.4f"|format(metric.f1_score) }}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Precision</div>
                            <div class="metric-value">{{ "%.4f"|format(metric.precision) }}</div>
                        </div>
                        <div class="metric-box">
                            <div class="metric-label">Recall</div>
                            <div class="metric-value">{{ "%.4f"|format(metric.recall) }}</div>
                        </div>
                    </div>
                    
                    {% if metric.model_name in model_reports %}
                    <div class="model-report">
                        <p><strong>Classification Report:</strong></p>
                        <pre>{{ format_classification_report(model_reports[metric.model_name]) }}</pre>
                    </div>
                    {% endif %}
                    
                    {% if metric.model_name in model_figures and model_figures[metric.model_name].plots %}
                    <div class="figure-container">
                        {% for figure in model_figures[metric.model_name].plots %}
                        <div class="figure">
                            <img src="data:image/png;base64,{{ figure.data }}" alt="{{ figure.name }}">
                            <p class="figure-caption">{{ figure.name }}</p>
                        </div>
                        {% endfor %}
                    </div>
                    {% else %}
                    <p>No visualization plots available for this model.</p>
                    {% endif %}
                </div>
                {% endfor %}
            </div>
            
            {% set has_shap_plots = false %}
            {% for model_name, figures in model_figures.items() %}
                {% if figures.shap and figures.shap|length > 0 %}
                    {% set has_shap_plots = true %}
                {% endif %}
            {% endfor %}
            
            {% if shap_figures or has_shap_plots %}
            <div id="shap-analysis" class="section">
                <h2>SHAP Analysis</h2>
                
                {% if shap_figures %}
                <div class="subsection">
                    <h3>General SHAP Analysis</h3>
                    <div class="figure-container">
                        {% for figure in shap_figures %}
                        <div class="figure">
                            <img src="data:image/png;base64,{{ figure.data }}" alt="{{ figure.name }}">
                            <p class="figure-caption">{{ figure.name }}</p>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                {% endif %}
                
                {% for model_name, figures in model_figures.items() %}
                    {% if figures.shap and figures.shap|length > 0 %}
                    <div class="subsection">
                        <h3>SHAP Analysis for {{ model_name }}</h3>
                        <div class="figure-container">
                            {% for figure in figures.shap %}
                            <div class="figure">
                                <img src="data:image/png;base64,{{ figure.data }}" alt="{{ figure.name }}">
                                <p class="figure-caption">{{ figure.name }}</p>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                    {% endif %}
                {% endfor %}
                
                <div class="subsection">
                    <h3>Understanding SHAP Values</h3>
                    <p>SHAP values explain how each feature contributes to the model's predictions:</p>
                    <ul>
                        <li><strong>Summary Plot:</strong> Shows the distribution of feature impacts across all samples</li>
                        <li><strong>Bar Plot:</strong> Shows average impact of each feature</li>
                        <li><strong>Waterfall Plot:</strong> Shows contribution of each feature for a specific sample</li>
                        <li><strong>Force Plot:</strong> Interactive visualization of feature contributions</li>
                        <li><strong>Beeswarm Plot:</strong> Shows feature values and their impact on model output</li>
                    </ul>
                </div>
            </div>
            {% endif %}
        </body>
        </html>
        """
        
        # Function to format classification report for display
        def format_classification_report(report):
            formatted = []
            formatted.append(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
            formatted.append("-" * 60)
            
            # Add each class
            for class_name, metrics in report.items():
                if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                    continue
                formatted.append(f"{class_name:<15} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} {metrics['f1-score']:<10.2f} {metrics['support']:<10}")
            
            # Add summary rows
            formatted.append("-" * 60)
            formatted.append(f"{'accuracy':<15} {'':<10} {'':<10} {report['accuracy']:<10.2f} {'':<10}")
            formatted.append(f"{'macro avg':<15} {report['macro avg']['precision']:<10.2f} {report['macro avg']['recall']:<10.2f} {report['macro avg']['f1-score']:<10.2f} {report['macro avg']['support']:<10}")
            formatted.append(f"{'weighted avg':<15} {report['weighted avg']['precision']:<10.2f} {report['weighted avg']['recall']:<10.2f} {report['weighted avg']['f1-score']:<10.2f} {report['weighted avg']['support']:<10}")
            
            return "\n".join(formatted)
        
        # Check if we have any SHAP analyses
        has_shap_plots = any(len(figs.get('shap', [])) > 0 for figs in model_figures.values())
        
        # Render HTML
        template = Template(html_template)
        html_content = template.render(
            config=json.dumps(experiment_config, indent=4),
            metrics=metrics,
            model_figures=model_figures,
            overview_figures=overview_figures,
            shap_figures=shap_figures,
            model_reports=model_reports,
            format_classification_report=format_classification_report,
            best_model_name=best_model_name,
            has_shap_plots=has_shap_plots
        )
        
        # Save HTML report
        html_path = Path(report_dir) / 'report.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"Generated HTML report at {html_path}")
        
    except Exception as e:
        import traceback
        print(f"Error generating HTML report: {e}")
        traceback.print_exc()
        print(f"Generating simplified fallback report...")
        
        # Generate a minimal report in case of errors
        generate_minimal_report(report_dir, experiment_config, results)
        
def create_report_directory(base_dir="reports", experiment_name=None):
    """
    Create a directory structure for storing experiment reports
    
    Args:
        base_dir: Base directory for reports
        experiment_name: Name of the experiment (will use timestamp if None)
        
    Returns:
        report_dir: Path to the report directory
        subdirs: Dictionary of subdirectories
    """
    # Create experiment name with timestamp if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"experiment_{timestamp}"
    
    # Create main report directory
    report_dir = Path(base_dir) / experiment_name
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    subdirs = {
        'figures': report_dir / 'figures',
        'models': report_dir / 'models',
        'metrics': report_dir / 'metrics',
        'data': report_dir / 'data'
    }
    
    # Create each subdirectory
    for subdir in subdirs.values():
        subdir.mkdir(exist_ok=True)
    
    print(f"Created report directory: {report_dir}")
    return report_dir, subdirs  
