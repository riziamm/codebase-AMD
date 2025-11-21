# Configurable Machine & Deep Learning Pipeline for Classification

This repository contains the official code for the paper submitted to JAMIA, titled:
**"Interpretable Multimodal Deep Learning and Clinician-Friendly Visualization for Non-Invasive Ocular Biomarkers in Early Age-Related Macular~Degeneration"**

It includes two main components:
  * A configurable **Machine Learning (ML) pipeline** (in the `src/` folder) for end-to-end experiments on tabular data.
  * A **Deep Learning (DL) pipeline** (`dl_pipeline_gen.py`) for multimodal analysis.

## Setup

#### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

#### 2. Create Environment and Install Dependencies
This project was tested with Python 3.8+. 
Install the dependencies:
```bash
# Create a virtual environment (Optional but recommended) 
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```
#### 3. Data Structure
Place your datasets in a data/ folder as follows:
```
/repository-root/
├── data/
│   ├── mpod.csv                 # Private dataset
│   ├── breast_cancer_data.csv   # Public dataset
│   ├── aps_training.csv         # Public dataset
│   └── my_raw_data.csv          # Your ML pipeline data
├── src/                         # ML Pipeline source code
│   ├── main.py
│   └── ...
├── scripts/                     # ML Pipeline helper scripts
│   └── 1_create_holdout.sh
├── reports                      # All reports saved here
├── dl_pipeline_gen.py           # DL Pipeline script
├── requirements.txt
└── README.md
```

## Machine Learning Pipeline
This pipeline is designed for end-to-end ML experiments on tabular data, from data splitting to batch training and evaluation.

### ML Workflow
#### Phase 1: Project Setup (ML Pipeline Only)
This one-time setup creates the batch_config_example.json template.
```bash
python src/setup.py
```
#### Phase 2: Data Preparation
This splits your raw data into train_...csv (for training) and test_...csv (for holdout evaluation).
```bash
python -m src.main --mode create_test_set \
  --data_path data/my_raw_data.csv \
  --test_size 0.2 \
  --output_dir data
```
#### Phase 3: Training (Batch Experiments)
This runs all experiments defined in your config file on the train_...csv data. All models, plots, and results are saved to a new timestamped folder in reports/.
```bash
# 1. Edit batch_config_example.json to define your experiments
# 2. Run the batch training:
python -m src.main --mode batch \
  --data_path data/train_my_raw_data.csv \
  --config_file batch_config_example.json \
  --report_dir reports/batch_experiments
```
#### Phase 4: Post-Training Aggregation (Optional)
This generates a detailed HTML report for the batch run you just completed.
```bash
python src/reporting/batch_report_detailed.py \
  --batch_dir reports/batch_experiments/batch_[TIMESTAMP]
  ```
#### Phase 5: Evaluation on Holdout Set
This runs your trained models against the prepared holdout data (test_test.pkl) and saves the final evaluation.
```bash
python -m src.main --mode evaluate \
  --model_paths reports/batch_experiments/batch_[TIMESTAMP]/experiment_42/models/*.pkl \
  --eval_data_path reports/batch_experiments/batch_[TIMESTAMP]/experiment_42/data/test_test.pkl \
  --report_dir reports/final_evaluation
```


## Deep Learning 
This pipeline supports training, tuning, and explaining several deep learning architectures mentioned in the paper.

### DL Features

* **Multiple Architectures:** Supports `CNN`, `Transformer`, and hybrid `CNNTransformer` (sequential and parallel) models.
* **Data Handling:** Automated preprocessing for datasets like breast_cancer, scania_aps, and mpod.

    * Includes imputation, scaling, and resampling (SMOTEENN/SMOTETomek).
    * Note: The mpod dataset is our curated human dataset and is not publicly available for ethical purposes. The other two toy datasets are available online.

* **Hyperparameter Tuning:** Includes parallelized grid search to optimize model parameters.
* **Robust Training:** Uses k-fold cross-validation for model training and selection.
* **Comprehensive Evaluation:**
    * Generates standard metrics (Accuracy, F1, Precision, Recall, AUC, Kappa).
    * Calculates 95% confidence intervals for F1 scores.
    * Performs statistical comparison between models using McNemar's test.
    * Saves all plots (ROC curves, confusion matrices) and results.
* **Explainability (XAI):**
    * Integrates SHAP (KernelExplainer) analysis for test and holdout sets.
    * Includes error analysis on misclassified instances.

### How to Run
The pipeline is controlled via command-line arguments.

* Key Arguments

    * --dataset: The dataset identifier to use (breast_cancer, scania_aps, mpod).
    * --data_path: Path to the training CSV file.
    * --test_data_path: Path to the test CSV (required for scania_aps).
    * --experiment_name: A unique name for the output report directory.
    * --models_to_run: A space-separated list of models to train (e.g., CNN, Transformer).
    * --tune_hyperparameters: A flag to enable grid search tuning.
    * --resampling_method: (Optional) SMOTEENN or SMOTETomek for imbalanced data.
    * --use_gpu: A flag to enable CUDA for training.

#### Example Commands
1. Run all models on the Breast Cancer dataset (with tuning):
```bash
python dl_pipeline_gen.py \
    --dataset breast_cancer \
    --data_path "data/breast_cancer_data.csv" \
    --experiment_name "BC_all_models_tuned" \
    --tune_hyperparameters \
    --use_gpu \
    --epochs_training 10 \
    --epochs_tuning 5
```

2. Run a single model (CNN) on the Scania APS dataset with resampling (no tuning):
```bash
python dl_pipeline_gen.py \
    --dataset scania_aps \
    --data_path "data/aps_training.csv" \
    --test_data_path "data/aps_test.csv" \
    --experiment_name "Scania_CNN_resampled" \
    --models_to_run CNN \
    --resampling_method SMOTETomek \
    --use_gpu \
    --epochs_training 10
```
### Output
All results are saved in the `reports/dl_pipeline/` directory, organized by your `--experiment_name`
```
reports/dl_pipeline/[experiment_name]/
├── data/               # Saved scaler, label encoder, and holdout data
├── figures/            # All plots (ROC, CM, SHAP)
├── metrics/            # All metrics (CSVs, JSONs)
├── models/             # Saved model weights (.pt)
├── post_analysis/      # Holdout set results and final summary
│   ├── figures/
│   └── metrics/
│       └── [experiment_name]_final_summary.csv  <-- Final results table
└── experiment_config.json
```

# Citation
If you use this code in your research, please cite our paper:
```
Rizia, M. M., van Kleef, J. P., Rai, B. B., Maddess, T., & Suominen, H. (2025). Advancing interpretable multimodal deep learning, clinically-valid information visualization, and non-invasive ocular biomarkers of age-related macular degeneration at early stages [Manuscript submitted for publication]. Journal of the American Medical Informatics Association.
```

# License
This project is licensed under the MIT License.
