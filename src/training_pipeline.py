from pathlib import Path
import matplotlib
# Force matplotlib to not use any Xwindows backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import shutil
import os
import json
import sys
from sklearn.metrics import classification_report #accuracy_score
# from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import train_test_split
import logging
logging.getLogger('shap').setLevel(logging.WARNING)
import traceback


def run_classification_pipeline_with_reporting(data_path, report_dir=None, **kwargs):
    """
    Run classification pipeline with reporting capabilities
    
    Args:
        data_path: Path to data
        report_dir: Directory to save reports (default: None)
        **kwargs: Additional parameters for run_classification_pipeline
        
    Returns:
        best_model: Best model
        results: Dictionary of results
    """
    from .core_logic import (
        run_classification_pipeline, prepare_data, set_seeds, 
        train_evaluate_model, save_best_model, plot_class_distribution,
        analyze_feature_group_importance, run_shap_analysis, plot_learning_curve, analyze_feature_group_zones
    )
    
    from .reporting.utils import (
    create_report_directory, save_experiment_config, save_model_metrics,
    create_visualization_plots, generate_html_report, save_figure
    )
    
    # Import model evaluation functions
    from .evaluation import save_test_data
    
    # Set random seeds for reproducibility
    set_seeds(42)
    
    # Create report directory if not provided
    if report_dir is None:
        report_dir, subdirs = create_report_directory()
    else:
        # Ensure report_dir is a Path object
        report_dir = Path(report_dir)
        subdirs = {
            'figures': report_dir / 'figures',
            'models': report_dir / 'models',
            'metrics': report_dir / 'metrics',
            'data': report_dir / 'data'
        }
        # Create subdirectories if they don't exist
        for subdir in subdirs.values():
            subdir.mkdir(exist_ok=True)
    
    # Save experiment configuration
    config = kwargs.copy()
    config['data_path'] = data_path
    save_experiment_config(config, report_dir)
    
    # Extract parameters from kwargs
    normalization = kwargs.get('normalization', 'standard')
    sampling_method = kwargs.get('sampling_method', 'none')
    is_binary = kwargs.get('is_binary', True)
    preserve_zones = kwargs.get('preserve_zones', True)
    sort_features = kwargs.get('sort_features', 'none')
    feature_indices = kwargs.get('feature_indices', None)
    selected_features = kwargs.get('selected_features', None)
    tune_hyperparams = kwargs.get('tune_hyperparams', False)
    analyze_shap = kwargs.get('analyze_shap', True)
    analyze_minkowski_dist = kwargs.get('analyze_minkowski_dist', False)
    transform_features = kwargs.get('transform_features', False)
    
    # Load data
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    # Select features if specified
    if selected_features is not None:
        data = data[selected_features + [data.columns[-1]]]
    elif feature_indices is not None:
        # Select only specified feature indices (each with 20 columns)
        selected_columns = []
        for idx in feature_indices:
            start_col = idx * 20
            end_col = start_col + 20
            selected_columns.extend(list(range(start_col, end_col)))
        
        # Add the target column (last column)
        selected_columns.append(data.shape[1] - 1)
        
        # Filter the dataframe
        data = data.iloc[:, selected_columns]
        print(f"Selected features: {feature_indices}, total columns: {len(selected_columns)}")
    
    # Prepare data
    print(f"Preparing data (binary={is_binary}, preserve_zones={preserve_zones}, sort_features={sort_features}, normalization={normalization})...")
    X, y, le, scaler, feature_names, actual_group_metrics = prepare_data(
        data, 
        normalization=normalization, 
        is_binary=is_binary,
        preserve_zones=preserve_zones,
        sort_features=sort_features,
        transform_features=transform_features
    )

    #    START >>  Save LE and Scaler   
    data_dir = subdirs['data'] 
    scaler_path = None

    if scaler is not None: 
        scaler_path = data_dir / 'scaler.pkl'
        try:
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Saved fitted Scaler to {scaler_path}")
        except Exception as e:
            print(f"Warning: Failed to save Scaler: {e}") # Keep error handling


    # Save LabelEncoder
    try:
        le_path = data_dir / 'label_encoder.pkl'
        with open(le_path, 'wb') as f:
            pickle.dump(le, f)
        print(f"Saved LabelEncoder to {le_path}")
    except Exception as e:
        print(f"Warning: Failed to save LabelEncoder: {e}")
    
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    # Save test data for later evaluation
    save_test_data(X_test, y_test, le, data_path, report_dir, feature_names)
    # IMPORTANT: Save the original y_train before any sampling is applied
    y_train_original = y_train.copy()   
    
    # To plot original class distribution
    # plt.figure(figsize=(8, 6))
    # plot_class_distribution(y_train_original, le, 'Original Class Distribution')
    # save_figure(plt.gcf(), "1_class_distribution_original", report_dir)
    # plt.close()
    
    # Apply sampling method if specified - ONLY to training data
    if sampling_method != 'none':
        print(f"{sampling_method} sampling applied to data")
        
        # After sampling is complete, create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Use the first subplot for original distribution
        plt.sca(ax1)  # Set current axis to the first subplot
        plot_class_distribution(y_train_original, le, 'Original Class Distribution')
        
        # Use the second subplot for distribution after sampling
        plt.sca(ax2)  # Set current axis to the second subplot
        plot_class_distribution(y_train, le, f'After {sampling_method.upper()} ({normalization.capitalize()})')
        
        plt.tight_layout()
        save_figure(plt.gcf(), f"1_class_distribution_comparison_{sampling_method}_{normalization}", report_dir)
        plt.close()
        
    
    
    # Run the original classification pipeline
    best_model, results = run_classification_pipeline(
        data_path=data_path,
        normalization=normalization,
        sampling_method=sampling_method,
        is_binary=is_binary,
        preserve_zones=preserve_zones,
        sort_features=sort_features,
        feature_indices=feature_indices,
        selected_features=selected_features,
        tune_hyperparams=tune_hyperparams,
        analyze_shap=analyze_shap,
        analyze_minkowski_dist=analyze_minkowski_dist,
        transform_features=transform_features, 
        report_dir=report_dir
    )
    
    
    #    START: Enhanced Overfitting Gap Calculation   
    print("\n--- Calculating Overfitting Gap   ")
    train_f1 = 0.0   
    # If it returns e.g., 'test_f1_score', use that key instead.
    # test_f1 = results.get('f1_score', None)
    raw_test_f1 = results.get('f1_score', 0.0)
    test_f1 = float(raw_test_f1) if raw_test_f1 is not None else 0.0

    overfitting_gap = 0.0
    best_model_name = results.get('best_model_name', None)
    best_model_instance = None
    X_train_eval = None
    y_train_eval = None
    can_calculate = False

    #    Step 1: Verify Availability of Required Data   
    print("  Verifying data required for gap calculation...")
    print(f"    Available keys in results: {list(results.keys())}")
    if best_model_name is None:
        print("    ERROR: 'best_model_name' not found in results.")
    elif test_f1 is None:
        print("    ERROR: Test F1 score ('f1_scores' key) not found or is None in results.")
    elif 'trained_models' not in results:
        print("    ERROR: 'trained_models' dictionary not found in results.")
    elif best_model_name not in results['trained_models']:
        print(f"    ERROR: Best model '{best_model_name}' not found within 'trained_models' keys: {list(results['trained_models'].keys())}")
    elif 'X_train' not in results:
        print("    ERROR: 'X_train' not found in results.")
    elif 'y_train' not in results:
        print("    ERROR: 'y_train' not found in results.")
    else:
        # All keys seem present, try retrieving data
        try:
            best_model_instance = results['trained_models'][best_model_name]
            X_train_eval = results['X_train']
            y_train_eval = results['y_train']

            #    Step 2: Validate Retrieved Data   
            print("    Data retrieved. Validating contents...")
            if best_model_instance is None:
                 print("    ERROR: Retrieved best_model_instance is None.")
            elif X_train_eval is None:
                 print("    ERROR: Retrieved X_train_eval is None.")
            elif y_train_eval is None:
                 print("    ERROR: Retrieved y_train_eval is None.")
            # Check shapes only if data is not None
            elif not hasattr(X_train_eval, 'shape') or not hasattr(y_train_eval, 'shape'):
                 print("    ERROR: Retrieved X_train or y_train lacks shape attribute.")
            elif X_train_eval.shape[0] == 0:
                 print("    ERROR: Retrieved X_train_eval has 0 samples.")
            elif X_train_eval.shape[0] != y_train_eval.shape[0]:
                 print(f"    ERROR: Shape mismatch - X_train {X_train_eval.shape}, y_train {y_train_eval.shape}")
            else:
                 # If all checks pass
                 can_calculate = True
                 print("    Data validation successful.")
                 print(f"      Model Type: {type(best_model_instance)}")
                 print(f"      X_train shape: {X_train_eval.shape}")
                 print(f"      y_train shape: {y_train_eval.shape}, Unique labels: {np.unique(y_train_eval)}")

        except KeyError as ke:
            print(f"    ERROR retrieving data (KeyError): {ke}")
        except Exception as val_e:
            print(f"    ERROR during data retrieval/validation: {val_e}")
            traceback.print_exc()
    from sklearn.metrics import f1_score
    #    Step 3: Perform Calculation (only if possible)   
    if can_calculate:
        train_f1 = 0.0 # Initialize
        try:
            print(f"  Calculating training F1 for {best_model_name}...")
            y_train_pred = best_model_instance.predict(X_train_eval)
            print(f"    Prediction successful. y_train_pred shape: {y_train_pred.shape}, Unique: {np.unique(y_train_pred)}")

            # Check shapes before F1 calculation
            if y_train_eval.shape != y_train_pred.shape:
                 raise ValueError(f"Shape mismatch before F1 calc: y_train_eval {y_train_eval.shape}, y_train_pred {y_train_pred.shape}")

            train_f1 = f1_score(y_train_eval, y_train_pred, average='weighted', zero_division=0)
            overfitting_gap = max(0, train_f1 - test_f1) # Ensure gap is non-negative

            print(f"\n  Overfitting Analysis Results (Model: '{best_model_name}'):")
            print(f"    Training F1: {train_f1:.4f}")
            print(f"    Test F1:     {test_f1:.4f}")
            print(f"    Gap:         {overfitting_gap:.4f}")

            # Add warning flags
            if overfitting_gap > 0.2: print("    WARNING: Potential severe overfitting detected!")
            elif overfitting_gap > 0.1: print("    WARNING: Potential moderate overfitting detected.")

        except Exception as e:
            print(f"  ERROR calculating overfitting gap: {e}")
            traceback.print_exc()
            train_f1 = 0.0 # Reset on error
            overfitting_gap = 0.0 # Reset on error
    else:
        print("  Skipping overfitting gap calculation due to missing/invalid data or model. Gap defaults to 0.")
        train_f1 = 0.0 # Ensure defaults are set if skipped
        overfitting_gap = 0.0

    results['train_f1'] = train_f1
    results['test_f1'] = test_f1 
    results['overfitting_gap'] = overfitting_gap  
    
    trained_models = results.get('trained_models', {})
    accuracies = results.get('accuracies', {})
    f1_scores = results.get('f1_scores', {})
    roc_auc_scores = results.get('roc_auc_scores', {}) 
    avg_precision_scores = results.get('avg_precision_scores', {}) 
    
    # Find the best model name
    best_model_name = max(f1_scores, key=f1_scores.get) if f1_scores else "Unknown"
    
    # Sort models by F1 score to identify the best ones
    sorted_models = sorted([(name, f1_scores.get(name, 0)) for name in trained_models.keys()], key=lambda x: x[1], reverse=True)
    
    # Save top models
    models_dir = subdirs['models']
    top_models_saved = []
    for i, (model_name, f1_score) in enumerate(sorted_models):
        if i < 2:  # 0, 1, 2 = top 3
            model = trained_models[model_name]
            model_path = models_dir / f"{model_name.replace(' ', '_').lower()}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            top_models_saved.append(model_name)
            print(f"Saved {model_name} to {model_path} (Top {i+1} model, F1={f1_score:.4f})")
            
            if analyze_shap: 
                try:
                    print(f"\nAnalyzing feature group importance for {model_name} (Top {i+1} model)...")
                    analyze_feature_group_importance(
                        model, 
                        X_test, 
                        num_features=len(feature_indices) if feature_indices is not None else 9,
                        values_per_feature=20, 
                        metrics=metrics if 'metrics' in locals() else None,
                        class_index=0,
                        report_dir=report_dir,
                        y_test=y_test
                    )
                    
                except Exception as e:
                    print(f"Error analyzing feature group importance for {model_name}: {e}")
                    traceback.print_exc()            
            
    print(f"Saved top {len(top_models_saved)} models: {', '.join(top_models_saved)}")
    
    # Consolidate metrics into a single file
    all_metrics = {
        'best_model_name': best_model_name,
        'models': {}
    }

    if analyze_shap and best_model is not None:
        print(f"\nPerforming SHAP and Group Importance analysis on top models...")

        #    Define Feature Names and Grouping   
        metrics = ['mean', 'med', 'std', 'iqr', 'idr', 'skew', 'kurt', 'Del', 'Amp']
        num_zones = 20
        num_base_features = len(metrics)
        values_per_group_def = num_zones

 
        if feature_indices is not None:
            # Generate feature names only for the selected feature indices
            selected_feature_names = []
            for idx in feature_indices:
                if idx < len(metrics):  # Ensure idx is valid
                    metric = metrics[idx]
                    # Add all zones for this metric
                    for zone in range(1, num_zones + 1):
                        selected_feature_names.append(f'{metric}_Z{zone}')
                else:
                    print(f"Warning: Feature index {idx} exceeds metrics list length ({len(metrics)})")
            feature_names = selected_feature_names
        else:
            # Generate all feature names
            feature_names = [f'{m}_Z{i+1}' for m in metrics for i in range(num_zones)]
        
   
        if len(feature_names) != X_test.shape[1]:
            print(f"Warning: Generated feature names ({len(feature_names)}) mismatch X_test columns ({X_test.shape[1]}). Check generation logic.")
            # Fallback: Generate generic names if mismatch is critical
            feature_names = [f'feat_{i}' for i in range(X_test.shape[1])]
            
        num_models_to_analyze = min(3, len(sorted_models)) # Max 3 if needed

        #    Loop through top models   
        for i, (model_name, f1_score) in enumerate(sorted_models[:num_models_to_analyze]):
            print(f"\n--- Analyzing Model {i+1}/{num_models_to_analyze} {model_name}   ")
            model = trained_models[model_name]
            
            is_model_binary_from_config = kwargs.get('is_binary', True) 
            print(f"  DEBUG: Config 'is_binary' = {is_model_binary_from_config}") # Debug print

            class_indices_to_analyze = []
            # Determine number of classes based *primarily* on the config flag
            if is_model_binary_from_config:
                num_classes_for_shap = 2
                # print(f"  Setting num_classes_for_shap = 2 (Binary Mode)")
                class_indices_to_analyze = [1]
                print(f"    SHAP Class Analysis: Binary mode. Targeting positive class (index {class_indices_to_analyze}) for {num_classes_for_shap} classes.")
            else:
                num_classes_for_shap = 4
                # print(f"  Setting num_classes_for_shap = 4 (Multiclass Mode)")
                class_indices_to_analyze = list(range(num_classes_for_shap))
                print(f"  SHAP Class Analysis: Multiclass mode. Targeting {num_classes_for_shap} classes ({class_indices_to_analyze}).")
           
            
            # Define SHAP report directory for this specific model
            shap_report_dir = Path(report_dir) / f"shap_{model_name.replace(' ', '_').lower()}"
            shap_report_dir.mkdir(parents=True, exist_ok=True)
  
            base_model_check = model.steps[-1][1] if isinstance(model, Pipeline) else model
            is_tree_based = False
            if xgb and isinstance(base_model_check, xgb.XGBClassifier): is_tree_based = True
            if isinstance(base_model_check, (RandomForestClassifier, GradientBoostingClassifier)): is_tree_based = True

            try: 
                plot_types_standard = ['bar', 'summary']
                for plot_type in plot_types_standard:
                    run_shap_analysis(
                        model, X_test,
                        feature_names=feature_names,
                        plot_type=plot_type,
                        report_dir=report_dir,
                        num_feature_groups=num_base_features,
                        values_per_group=values_per_group_def,
                        group_metrics=metrics,
                        is_binary=is_binary,
                        y_test=y_test
                        
                    )
  
                sample_idx_to_plot = 0 
                plot_types_sample = ['waterfall', 'grouped_waterfall'] # Add the new grouped plot
                for plot_type in plot_types_sample:
                    run_shap_analysis(
                        model, X_test,
                        feature_names=feature_names,
                        sample_idx=sample_idx_to_plot,
                        plot_type=plot_type,
                        report_dir=report_dir,
                        num_feature_groups=num_base_features,
                        values_per_group=values_per_group_def,
                        group_metrics=metrics,
                        is_binary=is_binary,
                        y_test=y_test
                    )

                #    Run Advanced/Specific SHAP Plots (Conditional)   
                if is_tree_based: # Beeswarm often most insightful for trees
                    run_shap_analysis(
                        model, X_test,
                        feature_names=feature_names,
                        plot_type='beeswarm',
                        report_dir=report_dir,
                        num_feature_groups=num_base_features,
                        values_per_group=values_per_group_def,
                        group_metrics=metrics,
                        is_binary=is_binary,
                        y_test=y_test
                    ) 
                print(f"  Running class-specific analysis (Binary Mode: {is_model_binary_from_config})...")

                for class_idx in class_indices_to_analyze:
                    if class_idx >= num_classes_for_shap:
                        print(f"    Warning: Skipping class index {class_idx} as it exceeds determined number of classes ({num_classes_for_shap}).")
                        continue
                    print(f"    Analyzing Class Index {class_idx}...")
                    # Use a specific subdirectory for this class's SHAP results
                    class_shap_report_dir = shap_report_dir / f"class_{class_idx}"
                    class_shap_report_dir.mkdir(parents=True, exist_ok=True)

                    try:
                        #    Grouped Waterfall Plot   
                        print(f"      Generating grouped waterfall plot for sample 0, class {class_idx}...")
                        run_shap_analysis(
                            model, X_test,
                            feature_names=feature_names, # Pass feature names
                            sample_idx=0,
                            plot_type='grouped_waterfall',
                            class_index=class_idx, # Specify class for multiclass or positive class for binary
                            report_dir=class_shap_report_dir, # Save to class-specific dir
                            is_binary=is_model_binary_from_config, # Correctly pass binary status
                            y_test=y_test, # Pass y_test 
                            num_feature_groups=num_base_features,
                            values_per_group=values_per_group_def,
                            group_metrics=metrics
                        )

                        #    Feature Group Importance Analysis   
                        print(f"      Analyzing feature group importance for class {class_idx}...")
                        analyze_feature_group_importance(
                            model,
                            X_test,
                            num_features=num_base_features,
                            values_per_feature=values_per_group_def,
                            metrics=metrics,
                            class_index=class_idx, # Specify class
                            report_dir=class_shap_report_dir, # Save to class-specific dir
                            y_test=y_test 
                        )

                        #    Zone Importance Analysis   
                        # Example: Analyze 'skew' feature zones
                        print(f"      Analyzing zone importance for 'skew' features, class {class_idx}...")
                        analyze_feature_group_zones(model, X_test, feature_names, feature_indices=feature_indices,
                            group_name="skew", # Or loop through desired groups
                            report_dir=class_shap_report_dir, # Save to class-specific dir
                            class_index=class_idx, plot_type='bar')

                        # For heatmap visualization
                        analyze_feature_group_zones(model, X_test, feature_names, feature_indices=feature_indices,
                            group_name="skew", # Or loop through desired groups
                            report_dir=class_shap_report_dir, # Save to class-specific dir
                            class_index=class_idx, plot_type='heatmap')

                        # For clustered heatmap visualization
                        analyze_feature_group_zones(model, X_test, feature_names, feature_indices=feature_indices,
                            group_name="skew", # Or loop through desired groups
                            report_dir=class_shap_report_dir, # Save to class-specific dir
                            class_index=class_idx, plot_type='clustermap')

                    except Exception as e:
                        print(f"      !! Error generating plots for Class Index {class_idx} of {model_name}: {e}")
                        traceback.print_exc()

  

            except Exception as e:
                print(f"!! Error during analysis for model {model_name}: {e}")
                print(traceback.format_exc()) # Print full traceback for debugging
                print(f"!! Skipping remaining analysis for {model_name}")


        print("\n--- SHAP and Group Importance Analysis Complete   ")

    else:
        if not analyze_shap:
            print("\nSHAP analysis skipped (analyze_shap=False).")
        if best_model is None:
            print("\nSHAP analysis skipped (no best model identified).")
    
    
    # Generate visualization plots for each model and collect metrics
    for model_name, model in trained_models.items():
        # Generate detailed visualizations for each model
        create_visualization_plots(
            model, 
            X_train, X_test, y_train, y_test, 
            feature_names, 
            report_dir, 
            model_name, 
            le
        )
        
        # Generate detailed model results for all models (even if not saved)
        model_results = {
            'accuracy': accuracies.get(model_name, 0),
            'f1_score': f1_scores.get(model_name, 0),
            'roc_auc_score': roc_auc_scores.get(model_name, 0.0), 
            'avg_precision_score': avg_precision_scores.get(model_name, 0.0),
            'is_best_model': model_name == best_model_name,
            'classification_report': classification_report(
                y_test, 
                model.predict(X_test), 
                output_dict=True,
                zero_division=0
            )
        }
        
        # Extract precision and recall from classification report
        if 'classification_report' in model_results:
            if 'weighted avg' in model_results['classification_report']:
                model_results['precision'] = model_results['classification_report']['weighted avg']['precision']
                model_results['recall'] = model_results['classification_report']['weighted avg']['recall']
        
        all_metrics['models'][model_name] = model_results
    
    # Save consolidated metrics
    metrics_dir = subdirs['metrics']
    with open(metrics_dir / 'all_model_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Saved consolidated metrics to {metrics_dir / 'all_model_metrics.json'}")
    
    # Create model comparison visualization
    plt.figure(figsize=(12, 6))
    models = list(accuracies.keys())
    acc_values = [accuracies[m] for m in models]
    f1_values = [f1_scores[m] for m in models]
    roc_auc_values = [roc_auc_scores.get(m, 0.0) for m in models] 
    
    x = np.arange(len(models))
    width = 0.25
    
    plt.bar(x - width/2, acc_values, width, label='Accuracy')
    plt.bar(x + width/2, f1_values, width, label='F1 Score')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    save_figure(plt.gcf(), "2_model_comparison", report_dir)
    plt.close()
    
    
    
    # Generate comprehensive HTML report
    generate_html_report(report_dir, config, {
        'best_model_name': best_model_name,
        'best_model_accuracy': accuracies.get(best_model_name, 0),
        'best_model_f1': f1_scores.get(best_model_name, 0),
        'best_model_roc_auc': roc_auc_scores.get(best_model_name, 0.0), # 
        'best_model_avg_precision': avg_precision_scores.get(best_model_name, 0.0), 
        'all_models': trained_models,
        'accuracies': accuracies,
        'f1_scores': f1_scores
    })
    
    
    # Return best model and results dictionary with additional information
    return best_model, {
        'trained_models': trained_models,
        'accuracies': accuracies,
        'f1_scores': f1_scores,
        'roc_auc_scores': roc_auc_scores,
        'avg_precision_scores': avg_precision_scores,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'le': le,
        'best_model_name': best_model_name,
        'accuracy': accuracies.get(best_model_name, 0),
        'f1_score': f1_scores.get(best_model_name, 0),
        'report_dir': str(report_dir)
    }



def main():
    """
    Main function to run the classification pipeline with reporting or batch experiments
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run classification pipeline with reporting')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data file')
    parser.add_argument('--report_dir', type=str, help='Directory to save reports')
    parser.add_argument('--batch', action='store_true', help='Run batch experiments')
    parser.add_argument('--max_experiments', type=int, help='Maximum number of experiments to run in batch mode')
    parser.add_argument('--config_file', type=str, help='Path to JSON configuration file for batch experiments')
    
    # Add common classification pipeline parameters
    parser.add_argument('--normalization', type=str, default='standard', choices=['standard', 'minmax', 'robust', 'none'],
                        help='Normalization method')
    parser.add_argument('--sampling_method', type=str, default='none', choices=['none', 'smote', 'random_over', 'random_under'],
                        help='Sampling method for class imbalance')
    parser.add_argument('--binary', action='store_true', help='Use binary classification (default: False)')
    parser.add_argument('--preserve_zones', action='store_true', help='Preserve zone-wise structure (default: False)')
    parser.add_argument('--sort_features', type=str, default='none', 
                        choices=['none', 'ascend_all', 'descend_all', 'custom'],
                        help='Feature sorting strategy')
    parser.add_argument('--tune_hyperparams', action='store_true', help='Tune hyperparameters (default: False)')
    parser.add_argument('--analyze_shap', action='store_true', help='Perform SHAP analysis (default: False)')
    
    args = parser.parse_args()
    
    # Handle batch experiments
    if args.batch:
        from batch_experiments import run_batch_experiments, generate_experiment_configurations
        
        # Load configurations from file if provided
        if args.config_file:
            import json
            with open(args.config_file, 'r') as f:
                configs = json.load(f)
            
            # Run batch experiments
            best_model, summary = run_batch_experiments(
                data_path=args.data_path,
                configs=configs,
                max_experiments=args.max_experiments
            )
        else:
            # Define default parameter grid
            param_grid = {
                'normalization': ['standard', 'minmax', 'robust'],
                'sampling_method': ['none', 'smote'],
                'is_binary': [True, False],
                'preserve_zones': [True, False],
                'sort_features': ['none', 'ascend_all', 'descend_all'],
                'tune_hyperparams': [False]
            }
            
            # Run batch experiments
            best_model, summary = run_batch_experiments(
                data_path=args.data_path,
                param_grid=param_grid,
                max_experiments=args.max_experiments
            )
    else:
        # Run single classification pipeline with reporting
        best_model, results = run_classification_pipeline_with_reporting(
            data_path=args.data_path,
            report_dir=args.report_dir,
            normalization=args.normalization,
            sampling_method=args.sampling_method,
            is_binary=args.binary,
            preserve_zones=args.preserve_zones,
            sort_features=args.sort_features,
            tune_hyperparams=args.tune_hyperparams,
            analyze_shap=args.analyze_shap
            
        )
        
        print(f"\nBest model: {results['best_model_name']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"F1 Score: {results['f1_scores']:.4f}")
        print(f"Report saved to: {results['report_dir']}")

if __name__ == "__main__":
    main()