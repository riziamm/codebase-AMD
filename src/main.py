#!/usr/bin/env python3
"""
Machine Learning Classification Pipeline with Enhanced Reporting

This script serves as the main entry point for running classification experiments
with comprehensive reporting, visualization, and batch processing capabilities.

Added logging the output

1. Create holdout test set from raw data:
$ python -m src.main --mode create_test_set --data_path data/breast_cancer_data.csv --test_size 0.2 --output_dir data

2. Training
$ python -m src.main --mode batch \
  --data_path data/train_my_raw_data.csv \
  --config_file batch_config_example.json \
  --report_dir reports/batch_experiments
  
3. Evaluate a saved model:
$ python -m src.main --mode evaluate \
  --model_paths reports/batch_experiments/batch_[TIMESTAMP]/experiment_42/models/*.pkl \
  --eval_data_path reports/batch_experiments/batch_[TIMESTAMP]/experiment_42/data/test_test.pkl \
  --report_dir reports/final_evaluation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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

from .core_logic import preprocess_holdout_data
from .evaluation import evaluate_saved_model, compare_models, check_and_balance_test_data, create_holdout_test_set
from .training_pipeline import run_classification_pipeline_with_reporting
from .batch_runner import run_batch_experiments as run_batch_core
from .batch_runner import generate_experiment_configurations

logging.getLogger('shap').setLevel(logging.WARNING)

# Helper Function for JSON Serialization ---
def convert_to_json_serializable(obj):
    """Recursively converts numpy types and other non-serializable objects."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist() # Convert numpy arrays to lists
    elif isinstance(obj, (datetime, Path)): # Example: Convert datetime/Path to string
        return str(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    # Add other type conversions if needed
    return obj

def setup_environment():
    """Set up the environment by ensuring all directories exist"""
    # Create necessary directories
    directories = ['reports', 'models', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Add the current directory to the Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)

def run_single_experiment(args):
    """Run a single classification experiment with reporting"""
    
    # Addition: Load configuration from file if provided
    if args.config_file:
        import json
        try:
            with open(args.config_file, 'r') as f:
                config = json.load(f)
                print(f"Loaded configuration from {args.config_file}: {config}")
                
                # Update args with values from config
                for key, value in config.items():
                    if not hasattr(args, key) or getattr(args, key) is None:
                        setattr(args, key, value)
                    
                print(f"Updated arguments: {vars(args)}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            import traceback
            traceback.print_exc()
    
    # Check required arguments
    if not args.data_path:
        raise ValueError("--data_path is required for single experiment mode")
    
    # Run classification pipeline
    best_model, results = run_classification_pipeline_with_reporting(
        data_path=args.data_path,
        report_dir=args.report_dir,
        normalization=args.normalization,
        sampling_method=args.sampling_method,
        is_binary=args.binary,
        preserve_zones=args.preserve_zones,
        sort_features=args.sort_features,
        tune_hyperparams=args.tune_hyperparams,
        analyze_shap=args.analyze_shap,
        transform_features=args.transform_features
    )
    
    print(f"\nBest model: {results['best_model_name']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Report saved to: {results['report_dir']}")
    
    return best_model, results

def run_batch_experiments(args):
    """Run batch experiments with multiple configurations"""
    
    # Check required arguments
    if not args.data_path:
        raise ValueError("--data_path is required for batch experiment mode")
    
    # Load configurations from file if provided
    if args.config_file:
        with open(args.config_file, 'r') as f:
            configs = json.load(f)
        
        # Run batch experiments
        best_model, summary = run_batch_core(
            data_path=args.data_path,
            configs=configs,
            max_experiments=args.max_experiments,
            base_report_dir=args.report_dir or "reports/batch_experiments"
        )
    else:
        # Define default parameter grid
        param_grid = {
            'normalization': ['standard', 'minmax', 'robust'],
            'sampling_method': ['none', 'smote'],
            'is_binary': [True, False],
            'preserve_zones': [True, False],
            'sort_features': ['none', 'ascend_all', 'descend_all'],
            'tune_hyperparams': [False],
            'analyze_shap': [args.analyze_shap],
            'transform_features': [args.transform_features]
        }
        
        # Run batch experiments
        best_model, summary = run_batch_core(
            data_path=args.data_path,
            param_grid=param_grid,
            max_experiments=args.max_experiments,
            base_report_dir=args.report_dir or "reports/batch_experiments"
        )
    
    return best_model, summary

def evaluate_model(args):
    """Evaluate a saved model on test data"""
    if not args.eval_data_path:
        raise ValueError("--eval_data_path is required for evaluate mode (specify path to test_data.pkl or test_test.pkl)")
    if not Path(args.eval_data_path).is_file():
         raise FileNotFoundError(f"Evaluation data file not found: {args.eval_data_path}")

    # --- Load Evaluation Data and Print Summary ---
    eval_data_file_path = Path(args.eval_data_path)
    print("\n \n --------------------------------------------------------------")
    print(f"\n------------------ EVALUATING  ------------------")
    print(f"File: {eval_data_file_path}")
    eval_data = None
    data_summary = {} # <-- Initialize dictionary to store summary
    try:
        with open(eval_data_file_path, 'rb') as f:
            eval_data = pickle.load(f)
        print("Data loaded successfully.")

        if 'X_test' not in eval_data or 'y_test' not in eval_data:
             raise KeyError(...)

        y_eval = eval_data['y_test']
        total_samples = len(y_eval)
        class_counts = Counter(y_eval)
        sorted_classes = sorted(class_counts.keys())

        # --- Store Summary Info ---
        data_summary['file_path'] = str(eval_data_file_path)
        data_summary['total_samples'] = total_samples
        data_summary['num_classes'] = len(sorted_classes)
        data_summary['class_distribution'] = {}

        print(f"\n--- Evaluation Data Summary ---")
        print(f"Total Samples: {total_samples}")
        print(f"Number of Classes: {len(sorted_classes)}")
        print("Class Distribution:")

        le = eval_data.get('le', None)
        class_names_map = {} # To store mapping from encoded label to name
        if le:
             try:
                 class_names = le.inverse_transform(sorted_classes)
                 for class_idx, class_name in zip(sorted_classes, class_names):
                      count = class_counts[class_idx]
                      percentage = (count / total_samples) * 100
                      print(f"  Class '{class_name}' (Encoded: {class_idx}): {count} samples ({percentage:.1f}%)")
                      # Store detailed info
                      data_summary['class_distribution'][int(class_idx)] = { # Use int for JSON key
                          'name': class_name,
                          'count': count,
                          'percentage': round(percentage, 2)
                      }
                      class_names_map[int(class_idx)] = class_name
             except Exception as le_err:
                 print(f"  Warning: Could not use label encoder ({le_err}). Showing encoded labels only.")
                 le = None # Fallback

        if not le: # Print/Store with encoded labels only
             for class_idx in sorted_classes:
                 count = class_counts[class_idx]
                 percentage = (count / total_samples) * 100
                 class_label_str = str(class_idx) # Use string for JSON key if no name
                 print(f"  Class {class_idx}: {count} samples ({percentage:.1f}%)")
                 data_summary['class_distribution'][int(class_idx)] = { # Use int for JSON key
                     'name': f'Class_{class_idx}', # Generic name
                     'count': count,
                     'percentage': round(percentage, 2)
                 }
        print("-----------------------------")

    except FileNotFoundError:
        print(f"Error: Evaluation data file not found at {eval_data_file_path}")
        sys.exit(1)
    except KeyError as e:
         print(f"Error: Missing expected key {e} in evaluation data file {eval_data_file_path}.")
         print("Ensure the PKL file was saved with 'X_test' and 'y_test' keys.")
         sys.exit(1)
    except Exception as e:
        print(f"Error loading or summarizing evaluation data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    evaluation_results = {}
    if args.model_paths:
        # --- Compare multiple models ---
        print(f"Comparing models: {args.model_paths}")
        print(f"Using evaluation data: {args.eval_data_path}")

        # Check if balancing is requested for the evaluation data
        eval_data_for_comparison = args.eval_data_path
        if args.balance_test:
             print("Balancing evaluation data for comparison...")
             # Note: Balancing evaluation data can be misleading, use with caution.
             X_balanced, y_balanced = check_and_balance_test_data(
                 test_data_path=args.eval_data_path, # Load from the path
                 min_samples_per_class=args.min_samples_per_class,
                 balance_method=args.balance_method
             )
             # We need to pass the balanced data, not the path, to compare_models if balanced.
             # compare_models might need adjustment to accept X, y directly OR save balanced data to temp file.
             # For simplicity now, we will proceed WITHOUT balancing in compare mode if requested.
             # Consider modifying compare_models if balanced comparison is essential.
             print("WARNING: Balancing (--balance_test) is currently ignored in multi-model comparison mode.")
             # eval_data_for_comparison = (X_balanced, y_balanced) # Need compare_models to handle this

        results = compare_models(
            model_paths=args.model_paths,
            # Use eval_data_path for the primary test data
            test_data_path=eval_data_for_comparison, # Pass the path
            # Remove distinct holdout path concept from compare_models call for now
            # holdout_data_path=None,
            report_dir=args.report_dir,
            # Balancing handled above (currently warning issued)
            balance_test_data=False, # Let check_and_balance handle it if needed before call
            balance_method=args.balance_method,
            analyze_shap=args.analyze_shap
        )
        
        evaluation_results = results
        # return results
    else:
        # --- Evaluate single model ---
        if not args.model_path:
            raise ValueError("--model_path is required for single model evaluate mode")

        print(f"Evaluating single model: {args.model_path}")
        print(f"Using evaluation data: {args.eval_data_path}")

        # If balance_test is requested, first load, balance data
        if args.balance_test:
            print("Balancing evaluation data...")
            # Load, balance, and get the actual data arrays
            X_eval_balanced, y_eval_balanced = check_and_balance_test_data(
                test_data_path=args.eval_data_path, # Load from the path
                min_samples_per_class=args.min_samples_per_class,
                balance_method=args.balance_method
            )

            # Evaluate with the balanced data (pass X and y directly)
            results = evaluate_saved_model(
                model_path=args.model_path,
                # Pass the balanced data arrays directly
                X_test=X_eval_balanced,
                y_test=y_eval_balanced,
                # test_data_path=None, # Don't pass path if passing arrays
                # holdout_data_path=None, # Remove distinct holdout concept here
                report_dir=args.report_dir,
                analyze_shap=args.analyze_shap,
                class_balance_report=True # Indicate balancing was done
            )
        else:
            # Standard evaluation with original evaluation data path
            results = evaluate_saved_model(
                model_path=args.model_path,
                test_data_path=args.eval_data_path,
                # holdout_data_path=None, # Remove distinct holdout concept here
                report_dir=args.report_dir,
                analyze_shap=args.analyze_shap,
                class_balance_report=True # Report original balance
            )
            
        # Explicitly print the classification report string ***
        print("\n--- Evaluation Metrics ---")
        if results:
            # Print Accuracy and F1 Score (assuming they are in results)
            print(f"Accuracy: {results.get('accuracy', 'N/A'):.4f}")
            print(f"F1 Score (Weighted): {results.get('f1_score', 'N/A'):.4f}")

            # Print the Classification Report String
            report_string = results.get('classification_report_str')
            if report_string:
                print("\nClassification Report:")
                print(report_string)
            else:
                print("\nClassification Report: Not found in results.")
        else:
            print("Evaluation did not produce results.")
        # Assign results for potential later use
        evaluation_results = results
        
        
        # --- START FIX: JSON saving logic moved outside the if/else block ---
    if args.report_dir and evaluation_results:
        print(f"\n--- Saving Evaluation Metrics and Summary ---")
        # The report_dir from args is the specific evaluation output directory
        # e.g., .../evaluation_outputs/validation_test
        metrics_dir = Path(args.report_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        json_filename = 'comparison_metrics.json' if args.model_paths else 'evaluation_metrics.json'
        json_save_path = metrics_dir / json_filename

        # This dictionary will be saved to JSON. It includes both the data summary
        # and the full results dictionary from the evaluation.
        final_output_data = {
            'data_summary': data_summary,
            'evaluation_results': evaluation_results
        }

        try:
            # Use the helper function to ensure all data types (like numpy arrays) are serializable
            serializable_data = convert_to_json_serializable(final_output_data)
            with open(json_save_path, 'w') as f:
                json.dump(serializable_data, f, indent=4)
            print(f"SUCCESS: Saved evaluation metrics to: {json_save_path}")
        except Exception as e:
            print(f"FATAL: An unexpected error occurred while saving the evaluation JSON: {e}")
            traceback.print_exc()

    elif not args.report_dir:
        print("\nSkipping JSON results saving as --report_dir was not provided.")
    elif not evaluation_results:
        print("\nSkipping JSON results saving as evaluation failed or produced no results.")
    #  # --- END
        
        
        # --- --> didnt work>> Combine and Save Results to JSON ---
        # Use the 'results' variable which holds the actual evaluation output
        # if args.report_dir and results: # Check if report dir specified AND results exist
        #     print(f"\n--- Saving Evaluation Metrics and Summary ---")
        #     metrics_dir = Path(args.report_dir) / 'metrics'
        #     metrics_dir.mkdir(parents=True, exist_ok=True)
        #     # json_save_path = metrics_dir / 'evaluation_metrics.json'
        #     # ensure the correct filename is used for single vs. multiple models
        #     json_filename = 'comparison_metrics.json' if args.model_paths else 'evaluation_metrics.json'
        #     json_save_path = metrics_dir / json_filename
        #     # ---  ---
            

        #     serializable_results = convert_to_json_serializable(results)

        #     # Combine summary and the *serializable* results
        #     final_output_data = {
        #         'data_summary': data_summary, # data_summary is likely already serializable
        #         'evaluation_results': serializable_results # Use the cleaned results
        #     }

        #     # ***  Apply conversion again to the final dict for extra safety ***
        #     # Although applying to 'results' first is key, converting the final dict
        #     # handles potential non-serializable items added in data_summary (like Path objects if not handled).
        #     serializable_final_data = convert_to_json_serializable(final_output_data)
    
        #     try:
        #         with open(json_save_path, 'w') as f:
        #             # *** Use the fully serialized data for dumping ***
        #             json.dump(serializable_final_data, f, indent=4)
        #         print(f"Saved evaluation summary and metrics to: {json_save_path}")
        #     except TypeError as json_err:
        #         print(f"Error saving results to JSON: {json_err}")
        #         # Consider logging the specific non-serializable object if possible
        #         print("Attempting to log problematic data structure (might be large):")
        #         try:
        #             # Be cautious logging potentially large data
        #             print(str(serializable_final_data)[:1000] + "...")
        #         except Exception as log_err:
        #             print(f"Could not log data structure: {log_err}")
        #         import traceback
        #         traceback.print_exc()

        #     except Exception as e:
        #         print(f"Error saving evaluation JSON: {e}")
        #         import traceback
        #         traceback.print_exc()


        # elif not args.report_dir:
        #     print("\nSkipping JSON results saving as --report_dir was not provided.")
        # # This check now correctly reflects if 'results' is empty or failed
        # elif not results:
        #     print("\nSkipping JSON results saving as evaluation failed or produced no results.")        

        return evaluation_results #results
    
    # --- OLD----
    #     # Compare multiple models
    #     if not args.test_data_path:
    #         raise ValueError("--test_data_path is required when comparing models")
        
    #     results = compare_models(
    #         model_paths=args.model_paths,
    #         test_data_path=args.test_data_path,
    #         holdout_data_path=holdout_path, # Added holdout path
    #         report_dir=args.report_dir,
    #         # Note: Balancing holdout is usually discouraged. Add logic if needed.
    #         balance_test_data=args.balance_test if validation_path else False, # Only balance validation set maybe
    #         # balance_test_data=args.balance_test,
    #         balance_method=args.balance_method
    #     )
    #     return results
    # else:
    #     # Evaluate single model
    #     if not args.model_path:\
    #         raise ValueError("--model_path is required for single model evaluate mode")
    #     # ---  amendment -------------
    #     print(f"Evaluating single model: {args.model_path}")
    #     results = evaluate_saved_model(
    #         model_path=args.model_path,
    #         test_data_path=validation_path, # Pass validation path
    #         holdout_data_path=holdout_path, # Pass holdout path
    #         report_dir=args.report_dir,
    #         analyze_shap=args.analyze_shap
    #         # run_shap=args.run_shap # Pass the SHAP flag
    #     )
    #     # --- end amendment -------------
    #     if not args.test_data_path and not (args.data_path and args.report_dir):
    #         raise ValueError("Either --test_data_path or both --data_path and --report_dir are required")
        
    #     # If balance_test is requested, first load, balance, and replace test data
    #     if args.balance_test and args.test_data_path:
    #         X_test, y_test = check_and_balance_test_data(
    #             test_data_path=args.test_data_path,
    #             min_samples_per_class=args.min_samples_per_class,
    #             balance_method=args.balance_method
    #         )
            
    #         # Now evaluate with the balanced data
    #         results = evaluate_saved_model(
    #             model_path=args.model_path,
    #             X_test=X_test,
    #             y_test=y_test,
    #             report_dir=args.report_dir,
    #             class_balance_report=True
    #         )
    #     else:
    #         # Standard evaluation with original test data
    #         results = evaluate_saved_model(
    #             model_path=args.model_path,
    #             test_data_path=args.test_data_path,
    #             report_dir=args.report_dir,
    #             class_balance_report=True
    #         )
            
    #     return results

def create_test_set(args):
    """Create a separate holdout test set from the original data"""
    
    if not args.data_path:
        raise ValueError("--data_path is required for create_test_set mode")
    
    train_path, test_path = create_holdout_test_set(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state,
        output_dir=args.output_dir,
        stratify=True  # Always use stratification for class balance
    )
    
    print(f"Created train data: {train_path}")
    print(f"Created test data: {test_path}")
    
    return train_path, test_path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Enhanced ML Classification Pipeline')
    
    # Main operation modes
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['single', 'batch', 'evaluate', 'create_test_set', 'preprocess_holdout'], # <-- ADDED 
                        help='Mode of operation')
    
    # Data paths
    parser.add_argument('--data_path', type=str, help='Path to the data file')
    parser.add_argument('--model_path', type=str, help='Path to the saved model (for evaluate mode)')
    parser.add_argument('--eval_data_path', type=str, help='Path to evaluation data: either test_data.pkl (validation) or holdout.csv') # Combined argument
    parser.add_argument('--holdout_csv_path', type=str,
                       help='Path to the raw holdout CSV file (REQUIRED for preprocess_holdout mode)')
    parser.add_argument('--training_report_dir', type=str,
                       help='Path to the report directory of the TRAINING RUN whose artifacts should be used (REQUIRED for preprocess_holdout mode)')    
    
    # Output directories
    parser.add_argument('--report_dir', type=str, help='Directory to save reports')
    parser.add_argument('--output_dir', type=str, default='data', help='Directory to save split dataset')
    
    # Batch experiment options
    parser.add_argument('--config_file', type=str, help='Path to JSON configuration file for batch experiments')
    parser.add_argument('--max_experiments', type=int, help='Maximum number of experiments to run in batch mode')
    
    # Classification parameters
    parser.add_argument('--normalization', type=str, default='standard', 
                        choices=['standard', 'minmax', 'robust', 'none'],
                        help='Normalization method')
    parser.add_argument('--sampling_method', type=str, default='none', 
                        choices=['none', 'smote', 'random_over', 'random_under'],
                        help='Sampling method for class imbalance')
    parser.add_argument('--binary', action='store_true', 
                        help='Use binary classification (default: True)')
    parser.add_argument('--preserve_zones', action='store_true', 
                        help='Preserve zone-wise structure (default: False)')
    parser.add_argument('--sort_features', type=str, default='none', 
                        choices=['none', 'ascend_all', 'descend_all', 'custom'],
                        help='Feature sorting strategy')
    parser.add_argument('--tune_hyperparams', action='store_true', 
                        help='Tune hyperparameters (default: False)')
    parser.add_argument('--analyze_shap', action='store_true', 
                        help='Perform SHAP analysis (default: False)')
    parser.add_argument('--transform_features', action='store_true',
                        help='Apply feature transformations (default: False)')
    
    # Test set creation options
    parser.add_argument('--test_size', type=float, default=0.1,
                       help='Size of the test set when splitting data (default: 0.1)')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
                        
    # Test data balancing options
    parser.add_argument('--balance_test', action='store_true',
                        help='Balance test data for evaluation (default: True)')
    parser.add_argument('--balance_method', type=str, default='oversample',
                        choices=['undersample', 'oversample', 'none'],
                        help='Method to balance test data (default: oversample)')
    parser.add_argument('--min_samples_per_class', type=int, default=None,
                        help='Minimum samples per class for balanced test set (default: auto)')
    
    # Model comparison options
    parser.add_argument('--model_paths', type=str, nargs='+',
                       help='List of model paths to compare (for evaluate mode)')
    
    return parser.parse_args()

def main():
    """Main function to run the ML pipeline"""
    # Set up the environment
    setup_environment()
    # Parse command line arguments
    args = parse_arguments()
    
    # Execute the requested mode
    try:
        if args.mode == 'single':
            best_model, results = run_single_experiment(args)
            print("\nExperiment completed successfully!")
            
        elif args.mode == 'batch':
            best_model, summary = run_batch_experiments(args)
            print("\nBatch experiments completed successfully!")
            
        elif args.mode == 'evaluate':
            results = evaluate_model(args)
            print("\nModel evaluation completed successfully!")
            
        elif args.mode == 'create_test_set':
            train_path, test_path = create_test_set(args)
            print("\nTest set creation completed successfully!")
            
        # --- START: Add logic for preprocess_holdout mode ---
        elif args.mode == 'preprocess_holdout':
            print("\n--- Running Holdout Preprocessing Mode ---")
            if not args.holdout_csv_path or not args.training_report_dir:
                print("Error: --holdout_csv_path and --training_report_dir are required for preprocess_holdout mode.")
                sys.exit(1)

            # Define paths based on the TRAINING run directory
            training_report_path = Path(args.training_report_dir)
            config_path = training_report_path / 'experiment_config.json'
            encoder_path = training_report_path / 'data' / 'label_encoder.pkl'
            scaler_path = training_report_path / 'data' / 'scaler.pkl' # Path to the saved scaler

            # --- Load necessary files from the TRAINING run ---
            config = None
            loaded_le = None
            loaded_scaler = None
            df_holdout = None
            try:
                print(f"Loading training config from: {config_path}")
                with open(config_path, 'r') as f:
                    config = json.load(f)

                print(f"Loading LabelEncoder from: {encoder_path}")
                if not encoder_path.exists(): raise FileNotFoundError("LabelEncoder not found.")
                with open(encoder_path, 'rb') as f:
                    loaded_le = pickle.load(f)

                # Load scaler ONLY if normalization was used in training run
                if config.get('normalization', 'none') != 'none':
                    print(f"Loading Scaler from: {scaler_path} (Normalization: {config.get('normalization')})")
                    if not scaler_path.exists(): raise FileNotFoundError("Scaler specified in config but not found.")
                    with open(scaler_path, 'rb') as f:
                        loaded_scaler = pickle.load(f)
                else:
                    print("Scaler not needed (Normalization was 'none' in training config).")

                print(f"Loading raw holdout data from: {args.holdout_csv_path}")
                if not Path(args.holdout_csv_path).exists(): raise FileNotFoundError("Holdout CSV file not found.")
                df_holdout = pd.read_csv(args.holdout_csv_path)

            except FileNotFoundError as e:
                print(f"Error loading required file: {e}")
                print(f"Please ensure '{args.training_report_dir}' is a valid report directory containing config, encoder, and scaler (if needed).")
                sys.exit(1)
            except Exception as e:
                print(f"Error during file loading: {e}")
                traceback.print_exc()
                sys.exit(1)

            # --- Preprocess the holdout data ---
            X_holdout_processed, y_holdout_processed, feature_names = preprocess_holdout_data(
                df_holdout=df_holdout,
                config=config, 
                saved_le=loaded_le,
                saved_scaler=loaded_scaler # Pass the loaded scaler (or None)
            )

            # --- Save the processed data ---
            if X_holdout_processed is not None and y_holdout_processed is not None:
                training_report_path = Path(args.training_report_dir)
                output_dir = training_report_path / 'data'
                output_dir.mkdir(parents=True, exist_ok=True) # Ensure the data subdir exists
                output_path = output_dir / 'test_test.pkl' # Define the specific filename

                # Keep the rest of the saving logic the same
                processed_data = {
                    'X_test': X_holdout_processed,
                    'y_test': y_holdout_processed,
                    'feature_names': feature_names
                    # 'le': loaded_le # If evaluation code also needs 'le', load and save it too.            
                }
                try:
                    with open(output_path, 'wb') as f:
                        pickle.dump(processed_data, f)
                    # Update the print statement to show the correct path
                    print(f"\nSuccessfully saved processed holdout data to: {output_path}")
                except Exception as e:
                    print(f"Error saving processed holdout data: {e}")
                    traceback.print_exc()
                    sys.exit(1) # Exit if saving fails
            else:
                print("\nHoldout preprocessing failed. Output PKL file not saved.")
                sys.exit(1) # Exit if preprocessing failed

            print("\nHoldout data preprocessing completed successfully!")
        # --- END:
               
        else:
            print(f"Unknown mode: {args.mode}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    print("\nAll operations completed successfully!")

if __name__ == "__main__":
    main()