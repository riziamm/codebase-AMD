#!/bin/bash

# --- Configuration ---
PYTHON_EXECUTABLE="python" 
PYTHON_MODULE="src.main"    
MODE="preprocess_holdout"
HOLDOUT_CSV="data/test_mpod.csv" # Path to the holdout CSV. 

# --- List of Batch Directories to Process ---
# <<<!!! EDIT THIS LIST BELOW !!!>>>
# Put each path on a new line, enclosed in double quotes.
BATCH_DIRS=(
    "reports/batch_experiments/batch_20250523_163954"
    "reports/batch_experiments/batch_20250527_165720"
    # "path/to/your/other/batch_folder"
)
# <<<!!! END OF EDITABLE LIST !!!>>>

# --- Script Logic ---

echo "Starting experiment processing..."
echo "Python script: $PYTHON_SCRIPT"
echo "Mode: $MODE"
echo "Holdout CSV: $HOLDOUT_CSV"

# Check if the BATCH_DIRS array is empty
if [ ${#BATCH_DIRS[@]} -eq 0 ]; then
    echo "Warning: The BATCH_DIRS list in the script is empty. Please edit the script to add batch directory paths."
    exit 1
fi

echo "Processing ${#BATCH_DIRS[@]} batch directories defined in the script."
echo "====================**************************===================="

# Loop through all the batch directory paths defined in the BATCH_DIRS array
for batch_base_dir in "${BATCH_DIRS[@]}"; do
    echo ""
    echo "Processing Batch Directory: '$batch_base_dir'"

    # Check if the defined path is a valid directory
    if [ ! -d "$batch_base_dir" ]; then
        echo "  Warning: '$batch_base_dir' (defined in script) is not a valid directory. Skipping."
        echo "----------------------------------------"
        continue # Move to the next item in the array
    fi

    found_count=0
    while IFS= read -r -d $'\0' experiment_dir; do
        if [ -d "$experiment_dir" ]; then 
            echo "  Found Experiment: '$experiment_dir'"
            echo "    Executing: $PYTHON_EXECUTABLE $PYTHON_SCRIPT --mode $MODE --holdout_csv_path $HOLDOUT_CSV --training_report_dir \"$experiment_dir\""

            # --- Execute the Python Command ---
            "$PYTHON_EXECUTABLE" -m "$PYTHON_MODULE" --mode "$MODE" --holdout_csv_path "$HOLDOUT_CSV" --training_report_dir "$experiment_dir"
            # ----------------------------------

            echo "    ------------------------------------"
            found_count=$((found_count + 1))
        fi
    done < <(find "$batch_base_dir" -maxdepth 1 -type d -name 'experiment_*' -print0)

    if [ "$found_count" -eq 0 ]; then
         echo "  No 'experiment_*' subdirectories found in '$batch_base_dir'."
    fi
    echo "----------------------------------------"

done

echo ""
echo "========================================"
echo "Script finished."