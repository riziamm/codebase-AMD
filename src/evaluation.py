import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, auc, average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from datetime import datetime
import xgboost as xgb
import json
import logging

from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score, roc_auc_score
)
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import mannwhitneyu
from itertools import combinations

# Add necessary imports if they are missing
from .core_logic import preprocess_holdout_data, run_shap_analysis, run_shap_analysis_eval, get_standardized_model_name

# Configure a basic logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('shap').setLevel(logging.WARNING)


   # -------------Load data ------------     
        
def save_test_data(X_test, y_test, le, data_path, report_dir, feature_names=None):
    """
    Save test data for later evaluation
    
    Args:
        X_test: Test features
        y_test: Test labels
        le: Label encoder
        data_path: Original data path
        report_dir: Report directory
        feature_names: Names of features
    """
    data_dir = Path(report_dir) / 'data'
    
    # Analyze class balance
    class_counts = np.bincount(y_test)
    class_dist = {}
    
    print("\nClass distribution in saved test set:")
    for class_idx, count in enumerate(class_counts):
        class_name = class_idx
        if le is not None:
            try:
                class_name = le.inverse_transform([class_idx])[0]
            except:
                pass
        class_dist[str(class_name)] = int(count)
        print(f"Class {class_name}: {count} samples ({count/len(y_test)*100:.1f}%)")
    
    # Save test data
    with open(data_dir / 'test_data.pkl', 'wb') as f:
        pickle.dump({
            'X_test': X_test,
            'y_test': y_test,
            'le': le,
            'original_data_path': data_path,
            'feature_names': feature_names,
            'class_distribution': class_dist,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f)
    
    print(f"Saved test data to {data_dir / 'test_data.pkl'}")
    
    # Also save a summary as JSON for easier inspection
    try:
        import json
        summary = {
            'original_data_path': str(data_path),
            'test_samples': len(y_test),
            'feature_count': X_test.shape[1],
            'class_distribution': class_dist,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(data_dir / 'test_data_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save test data summary as JSON: {e}")

def load_test_data(report_dir):
    """
    Load test data from a report directory
    
    Args:
        report_dir: Report directory
    
    Returns:
        Dictionary with test data
    """
    data_dir = Path(report_dir) / 'data'
    
    # Load test data
    with open(data_dir / 'test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    return test_data

def load_model(model_path):
    """
    Load a saved model
    
    Args:
        model_path: Path to the model file
    
    Returns:
        Loaded model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

# -------------STATISTICAL ANALYSIS FUNCTIONS------------

def calculate_f1_ci_bootstrap(y_true, y_pred, n_bootstraps=1000, alpha=0.95):
    """Calculates the F1 score confidence interval using bootstrap."""
    f1_scores = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        indices = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue # Skip if a bootstrap sample has only one class
        f1 = f1_score(y_true[indices], y_pred[indices], average='weighted', zero_division=0)
        f1_scores.append(f1)
    
    lower_percentile = (1.0 - alpha) / 2.0 * 100
    upper_percentile = (alpha + (1.0 - alpha) / 2.0) * 100
    ci_lower = np.percentile(f1_scores, lower_percentile)
    ci_upper = np.percentile(f1_scores, upper_percentile)
    return ci_lower, ci_upper


def perform_mann_whitney_u_test(df, feature_column, target_column, class1_label=0, class2_label=1):
    """
    Performs the Mann-Whitney U test for a feature between two classes.

    Args:
        df (pd.DataFrame): The input dataframe.
        feature_column (str): The name of the feature column to test.
        target_column (str): The name of the target variable column.
        class1_label: The label for the first group.
        class2_label: The label for the second group.

    Returns:
        A tuple containing the U statistic and the p-value.
        
    # Example Usage (assuming raw data loaded in a DataFrame `data`):
        # target_col_name = data.columns[-1]
        # perform_mann_whitney_u_test(data, 'mean_Z1', target_col_name)
    """
    # Separate the data for the two classes
    group1 = df[df[target_column] == class1_label][feature_column]
    group2 = df[df[target_column] == class2_label][feature_column]

    if len(group1) > 0 and len(group2) > 0:
        # Perform the test
        u_statistic, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
        print(f"Mann-Whitney U Test for '{feature_column}':")
        print(f"  U-statistic: {u_statistic:.4f}")
        print(f"  P-value: {p_value:.4f}")
        return u_statistic, p_value
    else:
        print(f"Could not perform test for '{feature_column}'. Not enough data in one or both groups.")
        return None, None

def calculate_cohens_d(df, feature_column, target_column, class1_label=0, class2_label=1):
    """
    Calculates Cohen's d for a feature between two classes.

    Args:
        df (pd.DataFrame): The input dataframe.
        feature_column (str): The name of the feature column to test.
        target_column (str): The name of the target variable column.
        class1_label: The label for the first group.
        class2_label: The label for the second group.

    Returns:
        The Cohen's d value, or None if data is insufficient.
        
        
    # Example Usage (assuming raw data loaded in a DataFrame `data`):
    # target_col_name = data.columns[-1]
    # calculate_cohens_d(data, 'mean_Z1', target_col_name)
    """
    group1 = df[df[target_column] == class1_label][feature_column]
    group2 = df[df[target_column] == class2_label][feature_column]

    if len(group1) < 2 or len(group2) < 2:
        print(f"Could not calculate Cohen's d for '{feature_column}'. Not enough data.")
        return None

    # Calculate means and sizes
    mean1, mean2 = np.mean(group1), np.mean(group2)
    n1, n2 = len(group1), len(group2)

    # Calculate pooled standard deviation
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    if pooled_std == 0:
        print(f"Could not calculate Cohen's d for '{feature_column}'. Pooled standard deviation is zero.")
        return None

    # Calculate Cohen's d
    d = (mean1 - mean2) / pooled_std
    print(f"Cohen's d for '{feature_column}': {d:.4f}")
    return d


# -------------Evaluation FUNCTIONS------------
def evaluate_saved_model(model_path, # Keep model_path
                         test_data_path=None, # Keep for backward compat/validation eval
                         holdout_data_path=None, # ADDED: Path to raw holdout CSV
                         report_dir=None,
                         class_balance_report=True,
                         analyze_shap=False): # ADDED: Flag to run SHAP
    """
    Evaluate a saved model on validation data (test_data.pkl) or holdout data (holdout_data_path).
    Includes optional SHAP analysis on the specified dataset.

    Args:
        model_path: Path to the model file (.pkl).
        test_data_path: Path to the preprocessed validation data file (test_data.pkl).
        holdout_data_path: Path to the raw holdout data file (.csv).
        report_dir: Directory to save evaluation results and plots.
        class_balance_report: Whether to report class balance metrics.
        analyze_shap: Whether to perform SHAP analysis.

    Returns:
        Dictionary with evaluation results.
    """
    model_path = Path(model_path)
    model_dir = model_path.parent # Directory containing the model file

    # Load model
    model = load_model(model_path)
    print(f"Loaded model from {model_path}")

    X_eval, y_eval, le, feature_names = None, None, None, None
    eval_data_source = ""

    # Determine which dataset to use for evaluation
    if holdout_data_path:
        print(f"Attempting evaluation on HOLD OUT data: {holdout_data_path}")
        eval_data_source = "Holdout"
        holdout_df = pd.read_csv(holdout_data_path)

        # --- Load corresponding config, le, scaler ---
        config_path = model_dir.parent / 'experiment_config.json' # Assumes config is one level up from model dir
        le_path = model_dir.parent / 'data' / 'label_encoder.pkl' # Assumes LE is in data subdir
        scaler_path = model_dir.parent / 'data' / 'scaler.pkl' # Assumes scaler is in data subdir

        config, saved_le, saved_scaler = None, None, None
        try:
            with open(config_path, 'r') as f: config = json.load(f)
            print(f"Loaded config from {config_path}")
        except Exception as e: print(f"ERROR loading config {config_path}: {e}")
        try:
            with open(le_path, 'rb') as f: saved_le = pickle.load(f)
            print(f"Loaded LabelEncoder from {le_path}")
        except Exception as e: print(f"ERROR loading LabelEncoder {le_path}: {e}")
        try:
            with open(scaler_path, 'rb') as f: saved_scaler = pickle.load(f)
            print(f"Loaded Scaler from {scaler_path}")
        except FileNotFoundError: print(f"Scaler not found at {scaler_path} (may not be needed if normalization='none').")
        except Exception as e: print(f"ERROR loading Scaler {scaler_path}: {e}")

        if config and saved_le:
            # Preprocess the holdout data
            X_eval, y_eval, feature_names = preprocess_holdout_data(
                holdout_df, config, saved_le, saved_scaler
            )
            le = saved_le # Use the loaded label encoder for reporting
            if X_eval is None: # Check if preprocessing failed
                 print("Holdout data preprocessing failed. Aborting evaluation.")
                 return {}
        else:
            print("ERROR: Missing config or LabelEncoder needed for holdout preprocessing. Aborting evaluation.")
            return {}

    elif test_data_path:
        print(f"Attempting evaluation on VALIDATION data: {test_data_path}")
        eval_data_source = "Validation"
        try:
            with open(test_data_path, 'rb') as f:
                test_data = pickle.load(f)
            X_eval = test_data['X_test']
            y_eval = test_data['y_test']
            le = test_data.get('le', None)
            feature_names = test_data.get('feature_names', None)
            print(f"Loaded validation data. X shape: {X_eval.shape}, y shape: {y_eval.shape}")
        except Exception as e:
            print(f"ERROR loading validation data {test_data_path}: {e}")
            return {}
    else:
        print("ERROR: No evaluation data specified (use --test_data_path or --holdout_data_path).")
        return {}

    # --- Proceed with evaluation using X_eval, y_eval ---
    if X_eval is None or y_eval is None:
         print("ERROR: Evaluation data (X or y) is None. Cannot proceed.")
         return {}

    print(f"\n--- Evaluating {model_path.stem} on {eval_data_source} data ---")

    # Analyze class balance in the evaluation set

    # Make predictions
    y_pred = model.predict(X_eval)

    # Calculate metrics
    accuracy = accuracy_score(y_eval, y_pred)
    f1 = f1_score(y_eval, y_pred, average='weighted')
    report_dict = classification_report(y_eval, y_pred, output_dict=True, zero_division=0) # Use dict, handle zero division
    report_str = classification_report(y_eval, y_pred, zero_division=0) # String version for printing
    
    # Added ROC AUC and Average Precision 
    roc_auc_eval = None
    avg_precision_eval = None
    if hasattr(model, 'predict_proba'): 
        y_proba_eval = model.predict_proba(X_eval) 
        if y_proba_eval.shape[1] == 2: # Binary 
            y_proba_pos_eval = y_proba_eval[:, 1] 
            try:
                roc_auc_eval = roc_auc_score(y_eval, y_proba_pos_eval) 
                avg_precision_eval = average_precision_score(y_eval, y_proba_pos_eval) 
            except Exception as e_auc:
                print(f"Could not calculate binary AUC/AP: {e_auc}")
        elif y_proba_eval.shape[1] > 2: # Multiclass
            try:
                roc_auc_eval = roc_auc_score(y_eval, y_proba_eval, multi_class='ovr', average='macro')
                # AP for multiclass: Can be calculated per class then averaged.
                # For now, we'll focus on ROC AUC for multiclass detailed metric.
            except Exception as e_auc_mc:
                print(f"Could not calculate multiclass ROC AUC: {e_auc_mc}")

    # Print results
    print(f"Evaluation Results ({eval_data_source}):") #
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    if roc_auc_eval is not None: print(f"ROC AUC: {roc_auc_eval:.4f}")
    if avg_precision_eval is not None: print(f"Average Precision: {avg_precision_eval:.4f}")
    
    print("\nClassification Report:") #
    print(report_str) #

    # Setup report directory for evaluation results
    eval_report_dir = None
    if report_dir:
        eval_report_dir = Path(report_dir) / f"eval_{model_path.stem}_{eval_data_source.lower()}"
        eval_report_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving evaluation artifacts to: {eval_report_dir}")

        # Save metrics
        metrics = {
            'model_path': str(model_path),
            'evaluation_data_source': eval_data_source,
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc_eval, # Added
            'average_precision': avg_precision_eval,
            'classification_report': report_dict,
            # Add class distribution if needed
        }
        metrics_path = eval_report_dir / 'evaluation_metrics.json'
        try:
             with open(metrics_path, 'w') as f:
                 json.dump(metrics, f, indent=4)
             print(f"Saved metrics to {metrics_path}")
        except Exception as e: print(f"Error saving metrics: {e}")

    # Generate confusion matrix (using y_eval, y_pred, le)
    cm = confusion_matrix(y_eval, y_pred)
    plt.figure(figsize=(8, 6))
    
    # Use class names if available
    if le is not None:
        try:
            class_names = le.classes_
        except:
            class_names = [str(i) for i in range(len(np.unique(y_eval)))]
    else:
        class_names = [str(i) for i in range(len(np.unique(y_eval)))]
    
    # Normalize confusion matrix to show percentages (helps with imbalanced classes)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot both raw and normalized confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Confusion Matrix (counts)')
    
    # Normalized (percentages)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix (normalized)')
    
    plt.tight_layout()
    
    # Save results if report directory is provided
    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report_dict,
            'class_distribution': {str(i): int(count) for i, count in enumerate(class_names)}
        }
        with open(report_dir / 'evaluation_metrics.json', 'wb') as f:
            pickle.dump(metrics, f)
        plt.savefig(eval_report_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(plt.gcf())
    
    
    # If probability predictions are available and it's binary classification, plot ROC and PR curves
    if hasattr(model, 'predict_proba') and len(np.unique(y_eval)) == 2:
        y_prob = model.predict_proba(X_eval)[:, 1]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_eval, y_prob)
        roc_auc = roc_auc_score(y_eval, y_prob)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_eval, y_prob)
        avg_precision = average_precision_score(y_eval, y_prob)
        
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        
        plt.tight_layout()
        
        if report_dir is not None:
            plt.savefig(eval_report_dir/ 'roc_pr_curves.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        plt.close(plt.gcf())


    


    # --- Run SHAP analysis if requested ---
    if analyze_shap:
        print("\n--- Running SHAP Analysis ---")
        is_binary = len(np.unique(y_eval)) == 2
        print(f"Running SHAP analysis, detected {'binary' if is_binary else 'multiclass'} classification")
    
        if feature_names is None: # fix with relatable feature names/from column heading
            print("WARNING: Feature names not available for SHAP plots. Using generic names.")
            feature_names = [f'feature_{i}' for i in range(X_eval.shape[1])]
            # feature_names = list(holdout_df.columns[:-1]) if holdout_data_path else [f'feature_{i}' for i in range(X_eval.shape[1])]
            # if feature_idx is not None:
            #     feature_names = [feature_names[i] for i in feature_idx]
                
            

        # Run desired SHAP plots on the evaluated data (X_eval)
        run_shap_analysis(model, X_eval, feature_names=feature_names, plot_type='summary', report_dir=eval_report_dir, is_binary=is_binary,y_test=y_eval)
        run_shap_analysis(model, X_eval, feature_names=feature_names, plot_type='bar', report_dir=eval_report_dir, is_binary=is_binary,y_test=y_eval)
        # --- Add group waterfall and waterfall + circultar heatmap SHAP Plots ---
        sample_idx_to_plot = 0 # Plot for the first sample
        plot_types_sample = ['waterfall', 'grouped_waterfall'] # Add the new grouped plot
        for plot_type in plot_types_sample:
            run_shap_analysis(
                model, X_eval,
                feature_names=feature_names,
                sample_idx=sample_idx_to_plot,
                plot_type=plot_type,
                report_dir=eval_report_dir,
                # # Pass grouping parameters needed for grouped_waterfall
                # num_feature_groups=num_base_features,
                # values_per_group=values_per_group_def,
                group_metrics=metrics,
                is_binary=is_binary,
                y_test=y_eval
            )
        
        
            
        # elif plot_type == 'circular_heatmap':
        #     print("Preparing data for circular heatmap...")
        #     # This plot requires the base SHAP values for a specific class
        #     if values_for_plot is not None and group_metrics and 'values_per_group' in locals():
        #         plot_circular_shap_heatmap(
        #             shap_values=values_for_plot,
        #             feature_names=group_metrics,
        #             num_zones_per_metric=values_per_group,
        #             model_name_prefix=model_name_str,
        #             shap_output_dir=report_dir / 'figures' if report_dir else Path('.')
        #         )
        #     else:
        #         print("Skipping circular heatmap due to missing data (SHAP values, group_metrics, or values_per_group).")
        
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, xgb.XGBClassifier)):
             run_shap_analysis(model, X_eval, feature_names=feature_names, plot_type='beeswarm', report_dir=eval_report_dir, is_binary=is_binary,y_test=y_eval)
             
        # Add waterfall/force plots if desired
        run_shap_analysis(model, X_eval, feature_names=feature_names, sample_idx=0, plot_type='waterfall', report_dir=eval_report_dir, is_binary=is_binary,y_test=y_eval)

    plt.close('all') # Close all figures at the end

    return { # Return metrics dictionary
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc_eval, # added
        'average_precision': avg_precision_eval, # added
        'classification_report': report_dict,
        'evaluation_data_source': eval_data_source
        # Add other relevant info if needed
    }

def check_and_balance_test_data(test_data_path, min_samples_per_class=None, balance_method='undersample'):
    """
    Check test data for class balance and optionally balance it
    
    Args:
        test_data_path: Path to the test data
        min_samples_per_class: Minimum samples required for each class (if None, will be calculated)
        balance_method: Method to balance data ('undersample', 'oversample', or 'none')
    
    Returns:
        Balanced X_test and y_test, or the original if balance_method is 'none'
    """
    # Load test data
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    le = test_data.get('le', None)
    
    # Analyze class distribution
    unique_classes, class_counts = np.unique(y_test, return_counts=True)
    
    print("Current class distribution:")
    for class_label, count in zip(unique_classes, class_counts):
        class_name = class_label
        if le is not None:
            try:
                class_name = le.inverse_transform([class_label])[0]
            except:
                pass
        print(f"Class {class_name}: {count} samples ({count/len(y_test)*100:.1f}%)")
    
    # Check if balancing is needed
    if balance_method == 'none':
        print("No balancing requested, using original test data")
        return X_test, y_test
    
    # Determine minimum samples per class if not specified
    if min_samples_per_class is None:
        if balance_method == 'undersample':
            min_samples_per_class = min(class_counts)
            print(f"Using minimum sample count across classes: {min_samples_per_class}")
        else:
            min_samples_per_class = max(class_counts)
            print(f"Using maximum sample count across classes: {min_samples_per_class}")
    
    # Apply balancing
    if balance_method == 'undersample':
        # Random undersampling
        from imblearn.under_sampling import RandomUnderSampler
        sampler = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = sampler.fit_resample(X_test, y_test)
        method_name = "undersampling"
    else:
        # Random oversampling
        from imblearn.over_sampling import RandomOverSampler
        sampler = RandomOverSampler(random_state=42)
        X_balanced, y_balanced = sampler.fit_resample(X_test, y_test)
        method_name = "oversampling"
    
    # Report new class distribution
    unique_classes_balanced, class_counts_balanced = np.unique(y_balanced, return_counts=True)
    
    print(f"\nAfter {method_name}, new class distribution:")
    for class_label, count in zip(unique_classes_balanced, class_counts_balanced):
        class_name = class_label
        if le is not None:
            try:
                class_name = le.inverse_transform([class_label])[0]
            except:
                pass
        print(f"Class {class_name}: {count} samples ({count/len(y_balanced)*100:.1f}%)")
    
    return X_balanced, y_balanced

def compare_models_v1(model_paths, test_data_path, report_dir=None, balance_test_data=True, balance_method='undersample'):
    """
    Compare multiple saved models on the same test data
    
    Args:
        model_paths: List of paths to model files
        test_data_path: Path to the test data file
        report_dir: Directory to save comparison results
        balance_test_data: Whether to balance the test data for fair comparison
        balance_method: Method to balance data ('undersample', 'oversample', or 'none')
    """
    # Load test data
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    le = test_data.get('le', None)
    
    # Check and balance test data if requested
    if balance_test_data:
        X_test, y_test = check_and_balance_test_data(
            test_data_path, 
            balance_method=balance_method
        )
    
    # Evaluate each model
    results = {}
    for model_path in model_paths:
        model_name = Path(model_path).stem
        print(f"\nEvaluating model: {model_name}")
        
        # Load model
        model = load_model(model_path)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Calculate class-specific metrics
        class_metrics = {}
        for class_label in np.unique(y_test):
            mask = (y_test == class_label)
            class_acc = accuracy_score(y_test[mask], y_pred[mask])
            class_f1 = f1_score(y_test[mask], y_pred[mask], average='weighted')
            
            class_name = class_label
            if le is not None:
                try:
                    class_name = le.inverse_transform([class_label])[0]
                except:
                    pass
                
            class_metrics[str(class_name)] = {
                'accuracy': class_acc,
                'f1_score': class_f1,
                'support': int(mask.sum())
            }
        
        results[model_name] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'class_metrics': class_metrics,
            'y_pred': y_pred
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Print class-specific metrics
        print("\nClass-specific metrics:")
        for class_name, metrics in class_metrics.items():
            print(f"Class {class_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
    
    # Compare models with visualizations
    plt.figure(figsize=(12, 8))
    
    # Plot overall accuracy and F1 score
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    f1_scores = [results[model]['f1_score'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    plt.subplot(2, 1, 1)
    plt.bar(x - width/2, accuracies, width, label='Accuracy')
    plt.bar(x + width/2, f1_scores, width, label='F1 Score')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Overall Model Performance')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    
    # Plot class-specific F1 scores
    unique_classes = sorted(list(results[models[0]]['class_metrics'].keys()))
    
    plt.subplot(2, 1, 2)
    bar_width = width / (len(unique_classes) + 1)
    
    for i, class_name in enumerate(unique_classes):
        class_f1s = [results[model]['class_metrics'][class_name]['f1_score'] for model in models]
        plt.bar(x + (i - len(unique_classes)/2) * bar_width, class_f1s, bar_width, 
                label=f'Class {class_name}')
    
    plt.xlabel('Models')
    plt.ylabel('F1 Score')
    plt.title('Class-specific F1 Scores')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    
    # Save comparison results if report directory is provided
    if report_dir is not None:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(report_dir / 'model_comparison.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        # Save comparison plot
        plt.savefig(report_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(plt.gcf())
        
        # Create HTML report
        create_comparison_html_report(results, report_dir, balance_method if balance_test_data else 'none')
    
    # plt.show()
    # plt.close(plt.gcf())
    
    # Create confusion matrix comparison
    num_models = len(models)
    fig, axes = plt.subplots(1, num_models, figsize=(6*num_models, 5), squeeze=False)
    
    for i, model_name in enumerate(models):
        y_pred = results[model_name]['y_pred']
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        ax = axes[0, i]
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax)
        ax.set_title(f'{model_name}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    
    if report_dir is not None:
        plt.savefig(report_dir / 'confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    plt.close(plt.gcf())
    
    return results





    # return comparison_results


def compare_models(model_paths,
                   test_data_path=None,
                   holdout_data_path=None, # This argument is kept for signature consistency
                   report_dir=None,
                   analyze_shap=False,
                   **kwargs):
    """
    Compares multiple saved models on the same data, calculates comprehensive metrics,
    performs statistical tests, and runs a full SHAP analysis with correct feature names.
    """
    # Determine the evaluation data path
    eval_path = holdout_data_path if holdout_data_path else test_data_path
    if not eval_path:
        print("ERROR: No evaluation data path provided to compare_models.")
        return {}
        
    print(f"\nComparing models on data: {eval_path}")
    final_results = {'models': {}, 'statistical_tests': {}}

    # --- Step 1: Load Data and Experiment Config ONCE ---
    try:
        with open(eval_path, 'rb') as f:
            test_data = pickle.load(f)
        X_eval = test_data['X_test']
        y_eval = test_data['y_test']
        feature_names = test_data.get('feature_names', [f'feature_{i}' for i in range(X_eval.shape[1])])
    except Exception as e:
        print(f"FATAL: Could not load test data from {eval_path}. Error: {e}")
        return {}

    # Load the experiment configuration from the first model's directory.
    # This is crucial for getting the feature group names for SHAP plots.
    config = {}
    try:
        first_model_exp_dir = Path(model_paths[0]).parent.parent
        config_path = first_model_exp_dir / 'experiment_config.json'
        print(f"Loading experiment config for SHAP: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load config. SHAP group plots may be affected. Error: {e}")

    # --- Step 2: Loop Through Each Model to Evaluate and Run SHAP ---
    eval_data_source_name = "Holdout" if holdout_data_path else "Validation"
    print(f"\n--- Evaluating {len(model_paths)} models on {eval_data_source_name} data ---")
    
    for model_path_str in model_paths:
        model_path = Path(model_path_str)
        model = load_model(model_path)
        model_name = get_standardized_model_name(model)
        
        print(f"\n--->  Evaluating model: {model_name} <---")
        y_pred = model.predict(X_eval)

        # --- Metrics Calculation ---
        accuracy = accuracy_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred, average='weighted', zero_division=0)
        report_dict = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)
        f1_ci_low, f1_ci_upp = calculate_f1_ci_bootstrap(y_eval, y_pred)
        
        roc_auc, avg_precision = None, None
        if hasattr(model, 'predict_proba'):
            try:
                y_proba = model.predict_proba(X_eval)
                if len(np.unique(y_eval)) == 2:
                    roc_auc = roc_auc_score(y_eval, y_proba[:, 1])
                    avg_precision = average_precision_score(y_eval, y_proba[:, 1])
            except Exception as e_auc:
                print(f"  Warning: Could not calculate AUC/AP scores for {model_name}: {e_auc}")

        # Store all results for this model
        final_results['models'][model_name] = {
            'accuracy': accuracy, 'f1_score': f1, 'roc_auc': roc_auc,
            'average_precision': avg_precision, 'classification_report': report_dict,
            'f1_score_ci': [f1_ci_low, f1_ci_upp], 'y_pred': y_pred
        }
        
        print(f"  Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f} (95% CI: [{f1_ci_low:.4f}, {f1_ci_upp:.4f}])")

        # --- SHAP Analysis ---
        sample_indices_to_plot = [9]  # Indices of instances to plot SHAP values for
        if analyze_shap:
            shap_report_dir = Path(report_dir) / f"shap_{model_name}" if report_dir else None
            run_shap_analysis_eval(model, X_eval, feature_names, shap_report_dir, config, y_test=y_eval, sample_indices_to_plot=sample_indices_to_plot)



    # --- Step 3: Perform Pairwise Statistical Tests AFTER all models are evaluated ---
    model_names_for_stats = list(final_results['models'].keys())
    if len(model_names_for_stats) > 1:
        print("\n--- Performing Pairwise Statistical Comparison (McNemar's Test) ---")
        for model1_name, model2_name in combinations(model_names_for_stats, 2):
            y_pred1 = final_results['models'][model1_name]['y_pred']
            y_pred2 = final_results['models'][model2_name]['y_pred']
            
            # Contingency table for McNemar's test
            b = np.sum((y_pred1 == y_eval) & (y_pred2 != y_eval)) # Model 1 correct, Model 2 incorrect
            c = np.sum((y_pred1 != y_eval) & (y_pred2 == y_eval)) # Model 1 incorrect, Model 2 correct
            
            if b + c > 0: # Test is only informative if there are disagreements
                contingency_table = [[np.sum((y_pred1 == y_eval) & (y_pred2 == y_eval)), b],
                                     [c, np.sum((y_pred1 != y_eval) & (y_pred2 != y_eval))]]
                try:
                    result = mcnemar(contingency_table, exact=True)
                    p_value = result.pvalue
                    test_name = f"{model1_name}_vs_{model2_name}"
                    final_results['statistical_tests'].setdefault('mcnemar', {})[test_name] = {'p_value': p_value}
                    print(f"  - {model1_name} vs {model2_name}: p-value = {p_value:.4f}")
                except Exception as e_stat:
                    print(f"  - Could not run McNemar's test for {model1_name} vs {model2_name}: {e_stat}")
            else:
                print(f"  - {model1_name} vs {model2_name}: Models have identical predictions. No test run.")

    # Include the ground truth in the final output for reference
    final_results['y_true'] = y_eval.tolist() if isinstance(y_eval, np.ndarray) else y_eval
    return final_results

def create_comparison_html_report(results, report_dir, balance_method):
    """
    Create an HTML report comparing multiple models
    
    Args:
        results: Dictionary with model results
        report_dir: Directory to save the report
        balance_method: Method used to balance the test data
    """
    from jinja2 import Template
    import base64
    
    # Create HTML template
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Comparison Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            .best-model { background-color: #e6ffe6; }
            .worst-model { background-color: #ffe6e6; }
            .figure { margin: 20px 0; text-align: center; }
            .figure img { max-width: 100%; }
        </style>
    </head>
    <body>
        <h1>Model Comparison Report</h1>
        
        <h2>Test Data Information</h2>
        <p>Balance method: {{ balance_method }}</p>
        
        <h2>Overall Performance</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>F1 Score</th>
            </tr>
            {% for model_name, metrics in results.items() %}
            <tr {% if loop.first %}class="best-model"{% endif %}>
                <td>{{ model_name }}</td>
                <td>{{ "%.4f"|format(metrics.accuracy) }}</td>
                <td>{{ "%.4f"|format(metrics.f1_score) }}</td>
            </tr>
            {% endfor %}
        </table>
        
        <h2>Class-Specific Metrics</h2>
        {% for class_name in class_names %}
        <h3>Class {{ class_name }}</h3>
        <table>
            <tr>
                <th>Model</th>
                <th>Accuracy</th>
                <th>F1 Score</th>
            </tr>
            {% for model_name, metrics in results.items() %}
            <tr>
                <td>{{ model_name }}</td>
                <td>{{ "%.4f"|format(metrics.class_metrics[class_name].accuracy) }}</td>
                <td>{{ "%.4f"|format(metrics.class_metrics[class_name].f1_score) }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endfor %}
        
        <h2>Visualizations</h2>
        <div class="figure">
            <h3>Overall Performance Comparison</h3>
            <img src="model_comparison.png" alt="Model Comparison">
        </div>
        
        <div class="figure">
            <h3>Confusion Matrix Comparison</h3>
            <img src="confusion_matrix_comparison.png" alt="Confusion Matrix Comparison">
        </div>
    </body>
    </html>
    """
    
    # Get list of class names (from first model's results)
    first_model = next(iter(results.values()))
    class_names = sorted(list(first_model['class_metrics'].keys()))
    
    # Sort models by F1 score
    sorted_results = {k: v for k, v in sorted(
        results.items(), 
        key=lambda item: item[1]['f1_score'], 
        reverse=True
    )}
    
    # Render HTML
    template = Template(html_template)
    html_content = template.render(
        results=sorted_results,
        class_names=class_names,
        balance_method=balance_method
    )
    
    # Save HTML report
    html_path = Path(report_dir) / 'model_comparison.html'
    with open(html_path, 'w') as f:
        f.write(html_content)
    
    print(f"Generated HTML comparison report at {html_path}")
    
def create_holdout_test_set(data_path, test_size=0.2, random_state=42, output_dir='data', stratify=True):
    """
    Create a separate holdout test set from the original data with class balancing
    
    Args:
        data_path: Path to the original data
        test_size: Size of the test set (default: 0.2)
        random_state: Random state for reproducibility
        output_dir: Directory to save the train and test data
        stratify: Whether to stratify the split to maintain class balance (default: True)
    
    Returns:
        train_path: Path to the training data
        test_path: Path to the test data
    """
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        data = pd.read_csv(data_path)
        print(f"Loaded data with shape: {data.shape}")
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        raise
    
    # Get target column (assumed to be the last column)
    target_col = data.columns[-1]
    print(f"Target column: {target_col}")
    
    # Print class distribution before splitting
    print("Original class distribution:")
    class_dist = data[target_col].value_counts().sort_index()
    for class_label, count in class_dist.items():
        print(f"Class {class_label}: {count}")
    
    # Split data with stratification to maintain class balance
    try:
        if stratify:
            train_data, test_data = train_test_split(
                data, 
                test_size=test_size, 
                random_state=random_state,
                stratify=data[target_col]  # This ensures class balance is maintained
            )
        else:
            train_data, test_data = train_test_split(
                data, 
                test_size=test_size, 
                random_state=random_state
            )
        print(f"Split successful. Train size: {train_data.shape}, Test size: {test_data.shape}")
    except Exception as e:
        print(f"ERROR: Split failed: {e}")
        raise
    
    # Print class distribution after splitting
    print("\nTrain set class distribution:")
    train_dist = train_data[target_col].value_counts().sort_index()
    for class_label, count in train_dist.items():
        print(f"Class {class_label}: {count}")
    
    print("\nTest set class distribution:")
    test_dist = test_data[target_col].value_counts().sort_index()
    for class_label, count in test_dist.items():
        print(f"Class {class_label}: {count}")
    
    # Save train and test data
    try:
        train_path = output_dir / f"train_{Path(data_path).name}"
        test_path = output_dir / f"test_{Path(data_path).name}"
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
    except Exception as e:
        print(f"ERROR: Failed to save data: {e}")
        raise
    
    print(f"\nCreated train data with {len(train_data)} samples: {train_path}")
    print(f"Created test data with {len(test_data)} samples: {test_path}")
    
    return str(train_path), str(test_path)