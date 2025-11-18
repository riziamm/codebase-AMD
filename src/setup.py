#!/usr/bin/env python3
"""
Setup script for Enhanced ML Classification Pipeline

This script sets up the necessary directories and dependencies for the ML pipeline.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def create_directories():
    """Create necessary directories for the pipeline"""
    directories = [
        'reports',
        'reports/figures',
        'reports/models',
        'reports/metrics',
        'reports/data',
        'models',
        'data'
    ]
    
    print("Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created {directory}")
    
    print("All directories created successfully!")

def create_config_template():
    """Create a template configuration file for batch experiments"""
    config = [
        {
            "normalization": "standard",
            "sampling_method": "smote",
            "is_binary": True,
            "preserve_zones": True,
            "sort_features": "none",
            "tune_hyperparams": False,
            "analyze_shap": True,
            "transform_features": False
        },
        {
            "normalization": "minmax",
            "sampling_method": "smote",
            "is_binary": True,
            "preserve_zones": True,
            "sort_features": "none",
            "tune_hyperparams": True,
            "analyze_shap": False,
            "transform_features": False
        },
        {
            "normalization": "standard",
            "sampling_method": "smote",
            "is_binary": False,
            "preserve_zones": True,
            "sort_features": "ascend_all",
            "tune_hyperparams": False,
            "analyze_shap": True,
            "transform_features": False
        }
    ]
    
    import json
    with open('batch_config_template.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("Created batch configuration template: batch_config_template.json")

def copy_module_files():
    """
    Copy the main module files to the current directory
    This function assumes the module files are in the same directory as this script
    """
    source_files = [
        'classification_ML_pt22_v2.py',
        'report_generation.py',
        'model_evaluation.py',
        'batch_experiments.py',
        'integrated_pipeline.py',
        'main.py'
    ]
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Ensuring module files are available...")
    for file in source_files:
        source_path = os.path.join(script_dir, file)
        
        # Check if the file exists
        if not os.path.exists(source_path):
            print(f"Warning: {file} not found in {script_dir}")
            continue
        
        # Copy the file to the current directory if not the same
        if os.path.abspath(os.getcwd()) != script_dir:
            import shutil
            shutil.copy2(source_path, os.getcwd())
            print(f"Copied {file} to {os.getcwd()}")
        else:
            print(f"{file} already in current directory")

def create_example_script():
    """Create an example script to run a simple experiment"""
    example_script = """#!/usr/bin/env python3
# Example script to run a simple classification experiment

import os
import sys

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the main module
from main import main

# Set up command line arguments
sys.argv = [
    'main.py',
    '--mode', 'single',
    '--data_path', 'data/your_data.csv',  # Replace with your data file
    '--normalization', 'standard',
    '--sampling_method', 'smote',
    '--binary',
    '--analyze_shap'
]

# Run the main function
if __name__ == "__main__":
    main()
"""
    
    with open('example.py', 'w') as f:
        f.write(example_script)
    
    print("Created example script: example.py")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Setup Enhanced ML Classification Pipeline')
    
    parser.add_argument('--all', action='store_true', help='Perform all setup steps')
    parser.add_argument('--dependencies', action='store_true', help='Install dependencies')
    parser.add_argument('--dirs', action='store_true', help='Create directories')
    parser.add_argument('--config', action='store_true', help='Create configuration template')
    parser.add_argument('--copy', action='store_true', help='Copy module files')
    parser.add_argument('--example', action='store_true', help='Create example script')
    
    args = parser.parse_args()
    
    # If no specific options are provided, perform all steps
    if not any([args.all, args.dependencies, args.dirs, args.config, args.copy, args.example]):
        args.all = True
    
    return args

def main():
    """Main function to run the setup"""
    print("Setting up Enhanced ML Classification Pipeline...")
    
    args = parse_arguments()
    
    if args.all or args.dirs:
        create_directories()
    
    if args.all or args.config:
        create_config_template()
    
    # if args.all or args.copy:
    #     copy_module_files()
    
    # if args.all or args.example:
    #     create_example_script()
    
    print("\nSetup completed successfully!")
    print("You can now run experiments using the main.py script.")
    print("Example usage:")
    print("  python -m src.main --mode create_test_set --data_path data/your_data.csv --test_size 0.2 --output_dir data")
    print("  python -m src.main --mode single --data_path your_data.csv --normalization standard --sampling_method smote --binary --report_dir reports/experiments")
    print("  python -m src.main --mode batch --data_path data/train_my_raw_data.csv --config_file batch_config_example.json   --report_dir reports/batch_experiments")
    
    print("  python -m src.main --mode evaluate --model_path models/your_model.pkl --test_data_path data/test_data.pkl")
    
    

if __name__ == "__main__":
    main()