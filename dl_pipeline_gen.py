"""
```
This script is part of a deep learning pipeline for classification tasks.
# Example usage from command line:

    CHoose GPU:
    CUDA_VISIBLE_DEVICES=1 # Select GPU 1, GPU 0 by default

    $ python dl_pipeline_gen.py --data_path "data/mpod.csv" --use_gpu --experiment_name "Binaryclass_run_all.debug"
    $ python dl_pipeline.py --data_path "data/mpod.csv" --tune_hyperparameters --use_gpu --experiment_name "Binaryclass_run_all.011" 2>&1 | tee -a "reports/log_binary_007_$(date +'%Y%m%d_%H%M%S').log"
    $ python dl_pipeline.py --data_path "data/mpod.csv" --use_gpu
    
    #** Generic toy data run
    python dl_pipeline_gen.py --dataset breast_cancer --data_path "data/breast_cancer_data.csv" --use_gpu --experiment_name "BC_all_001-test" 2>&1 | tee -a "reports/BC_all_001-test_$(date +'%Y%m%d_%H%M%S').log" 
    
    New addision>> --resampling_method: SMOTEENN or SMOTETomek
    #** Scania_aps data run
    CUDA_VISIBLE_DEVICES=1 \
    python dl_pipeline_gen.py \
    --dataset scania_aps \
    --data_path "data/aps_training.csv" \
    --test_data_path "data/aps_test.csv" \
    --experiment_name "Scania_run_CNN_001_test" \
    --use_gpu \
    --batch_size 128 \
    --epochs_training 5 \
    --models_to_run CNN \
    --resampling_method SMOTETomek \
    2>&1 | tee -a "reports/Scania_run_CNN_001_test_$(date +'%Y%m%d_%H%M%S').log" 
    
    #** default run all models
    $ python dl_pipeline.py --data_path "data/mpod.csv" --experiment_name "all_run" --models_to_run CNN Transformer CNNTransformer_sequential CNNTransformer_parallel --tune_hyperparameters --use_gpu --experiment_name "Binaryclass_run_all.01Jul25.006" 2>&1 | tee -a "reports/log_binary_all..01Jul25.006_$(date +'%Y%m%d_%H%M%S').log"
    
    # Debug run one model
    python dl_pipeline_gen_v2.py --data_path "data/mpod.csv"  --models_to_run CNNTransformer_parallel --use_gpu --batch_size 32 --epochs_training 5 --cv_splits_training 3 --shap_num_samples 50 --select_features 0 5 8 --experiment_name "Binaryclass_debug_run_cnntx.002" 2>&1 | tee -a "reports/log_Binaryclass_debug_run_cnntx.002_$(date +'%Y%m%d_%H%M%S').log"   

```

"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                            roc_curve, auc, confusion_matrix, classification_report,
                            f1_score, roc_auc_score)
from scipy.special import softmax # Use scipy's stable softmax
from sklearn.model_selection import ParameterGrid
from imblearn.combine import SMOTEENN, SMOTETomek
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors
import seaborn as sns
import math
from collections import Counter
import torch.multiprocessing as mp
import os
from datetime import datetime
import json
from pathlib import Path
import logging
import shap
import traceback
import joblib # for saving scaler/encoder
import argparse
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import chi2
from sklearn.impute import SimpleImputer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dl_classf.log", mode='w'), # Overwrite log file each run
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("DLClassification")
logging.getLogger('shap').setLevel(logging.WARNING)

# --- Utility Functions ---
def create_report_directory(base_dir="reports/dl", experiment_name=None):
    """
    Create a directory structure for storing experiment reports

    Args:
        base_dir: Base directory for reports
        experiment_name: Name of the experiment (will use timestamp if None)

    Returns:
        report_dir: Path to the report directory
        subdirs: Dictionary of subdirectories
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"dl_experiment_{timestamp}"

    report_dir = Path(base_dir) / experiment_name
    report_dir.mkdir(parents=True, exist_ok=True)

    subdirs = {
        'figures': report_dir / 'figures',
        'models': report_dir / 'models',
        'metrics': report_dir / 'metrics',
        'data': report_dir / 'data',
        'post_analysis': report_dir / 'post_analysis'
    }
    for subdir in subdirs.values():
        subdir.mkdir(exist_ok=True)

    logger.info(f"Created report directory: {report_dir}")
    return report_dir, subdirs

def update_summary_results(results_list, summary_file_path="reports/dl_pipeline/master_summary.csv"):
    """
    Appends a list of results to a master CSV file. Creates the file if it doesn't exist.

    Args:
        results_list (list): A list of dictionaries, where each dictionary is a row.
        summary_file_path (str): The path to the master CSV summary file.
    """
    summary_df = pd.DataFrame(results_list)
    file_path = Path(summary_file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if file_path.exists():
            # Append without writing header
            summary_df.to_csv(file_path, mode='a', header=False, index=False)
            logger.info(f"Appended results to master summary file: {file_path}")
        else:
            # Create new file with header
            summary_df.to_csv(file_path, mode='w', header=True, index=False)
            logger.info(f"Created new master summary file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to update summary results file: {e}")

def set_seeds(seed=42):
    """Set random seeds for reproducibility across all libraries"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Set random seed to {seed}")

def save_figure(fig, filename, report_subdir, dpi=300):
    """
    Save a matplotlib figure to a specified subdirectory within the report directory

    Args:
        fig: Matplotlib figure object
        filename: Name of the file (without extension)
        report_subdir: Path to the subdirectory (e.g., report_dir / 'figures')
        dpi: DPI for the saved figure
    """
    report_subdir.mkdir(parents=True, exist_ok=True) # Ensure subdir exists
    filepath = report_subdir / f"{filename}.png"
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved figure to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save figure {filepath}: {e}")
    plt.close(fig)


def save_experiment_config(config, report_dir):
    """
    Save experiment configuration to a JSON file

    Args:
        config: Experiment configuration dictionary
        report_dir: Report directory path
    """
    serializable_config = {}
    for key, value in config.items():
        if isinstance(value, np.ndarray):
            serializable_config[key] = value.tolist()
        elif isinstance(value, (torch.Tensor, nn.Module)):
            serializable_config[key] = str(value)
        elif isinstance(value, Path):
            serializable_config[key] = str(value)
        elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            serializable_config[key] = str(value)
        else:
            serializable_config[key] = value

    config_path = Path(report_dir) / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(serializable_config, f, indent=4)
    logger.info(f"Saved experiment configuration to {config_path}")

def save_model_metrics(results, report_dir, model_name, subfolder='metrics'):
    """
    Save model metrics to CSV and JSON files in a specified subfolder.

    Args:
        results: Dictionary of model results
        report_dir: Base report directory path
        model_name: Name of the model
        subfolder: Name of the subfolder within report_dir for metrics
    """
    metrics_dir = Path(report_dir) / subfolder
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        'model_name': model_name,
        'accuracy': results.get('accuracy', 0),
        'f1_score': results.get('f1', 0),
        'precision': results.get('precision', 0),
        'recall': results.get('recall', 0),
        'auc': results.get('auc', 0),
        'epoch': results.get('epoch', 0), # If applicable
        'val_loss': results.get('val_loss', float('inf')), # If applicable
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_dir / f"{model_name}_metrics.csv", index=False)

    # Save full results dictionary as JSON for more details if needed
    full_results_path = metrics_dir / f"{model_name}_full_results.json"
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, pd.DataFrame):
            serializable_results[key] = value.to_dict()
        elif isinstance(value, (Path, torch.Tensor, nn.Module)):
             serializable_results[key] = str(value)
        elif not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
            serializable_results[key] = str(value)
        else:
            serializable_results[key] = value

    with open(full_results_path, 'w') as f:
        json.dump(serializable_results, f, indent=4)


    y_true = results.get('y_true', None)
    y_pred = results.get('y_pred', None)

    if y_true is not None and y_pred is not None and len(y_true) > 0 and len(y_pred) > 0:
        try:
            class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            with open(metrics_dir / f"{model_name}_classification_report.json", 'w') as f:
                json.dump(class_report, f, indent=4)
        except ValueError as e:
            logger.warning(f"Could not generate classification report for {model_name}: {e}")


    logger.info(f"Saved metrics for {model_name} in {metrics_dir}")
    return metrics, metrics_dir


# --- 1. MODEL DEFINITIONS ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe = pe.transpose(0, 1) # Original: [max_len, 1, d_model] -> [1, max_len, d_model]
                                # For batch_first=True in TransformerEncoderLayer, input is (N, S, E)
                                # So pe should be broadcastable to this.
                                # If x is (N, S, E), then pe should be (1, S, E) or (S,E)
        self.register_buffer('pe', pe.squeeze(1)) # Shape [max_len, d_model]

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        # self.pe shape: (max_len, d_model)
        x = x + self.pe[:x.size(1), :]
        return self.dropout(x)

class CNNModel(nn.Module):
    def __init__(self, feature_dim, num_classes=1, cnn_units=64, dropout_rate=0.4):
        super(CNNModel, self).__init__()
        
        # Ensure cnn_units is an integer
        if isinstance(cnn_units, (list, tuple)):
            logger.warning(f"CNNModel received cnn_units as {type(cnn_units)} ({cnn_units}), taking first element.")
            _cnn_units = int(cnn_units[0])
        else:
            _cnn_units = int(cnn_units)
        
        self.feature_dim = feature_dim # This is sequence length for Conv1D if input is (N, C_in, L_in)
                                     # If input is (N, L_in) and C_in=1, feature_dim is L_in.

        # Input to Conv1d: (N, C_in, L_in)
        # Here, we assume input x will be (batch_size, 1, feature_dim)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=_cnn_units, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(_cnn_units)
        self.conv2 = nn.Conv1d(_cnn_units, _cnn_units * 2, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(_cnn_units * 2)

        self.use_pooling = feature_dim > 1 # Check if pooling makes sense
        current_len = feature_dim
        if self.use_pooling :
             self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
             current_len = current_len // 2 # After one pooling layer

        # Calculate CNN output dimension
        # After convs, shape is (batch_size, cnn_units*2, current_len)
        # cnn_output_dim = (cnn_units * 2) * current_len
        # ---Ensure the final output dimension is an integer ---
        cnn_output_dim = int((_cnn_units * 2) * current_len)

        self.fc1 = nn.Linear(cnn_output_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        # Assuming x has shape (batch_size, 1, feature_dim)
        # If x is (batch_size, feature_dim), it needs unsqueeze(1)
        if len(x.shape) == 2: # (batch_size, features)
            x = x.unsqueeze(1) # (batch_size, 1, features) - C_in = 1

        batch_size = x.size(0)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.batch_norm1(x)
        if self.use_pooling:
             x = self.pool(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batch_norm2(x)
        # No pooling after second conv in this setup, adjust if needed
        x = self.dropout(x)

        x = x.view(batch_size, -1) # Flatten

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class TransformerModel(nn.Module):
    def __init__(self, feature_dim, num_classes=1, transformer_dim=64, transformer_heads=4,
                 transformer_layers=2, dropout_rate=0.2):
        super(TransformerModel, self).__init__()
        # feature_dim is the number of input features per time step/sequence element.
        # transformer_dim is the model dimension (d_model).

        # --- Explicitly cast to integer types ---
        transformer_dim = int(transformer_dim)
        transformer_heads = int(transformer_heads)
        transformer_layers = int(transformer_layers)
        # -----

        # Project input features to transformer_dim
        self.projection = nn.Linear(feature_dim, transformer_dim)
        self.pos_encoder = PositionalEncoding(transformer_dim, dropout_rate)
        
        

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4, # Standard practice
            dropout=dropout_rate,
            batch_first=True # Important: input format (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=transformer_layers
        )

        self.fc1 = nn.Linear(transformer_dim, 64) # Output of transformer is (batch, seq, transformer_dim)
                                                 # take mean over seq_len
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Input x: (batch_size, num_features) if tabular, or (batch_size, seq_len, num_features) if sequence
        # For this model to make sense, we usually treat tabular data as a sequence of length 1.
        if len(x.shape) == 2: # (batch_size, features)
            x = x.unsqueeze(1)  # -> (batch_size, 1, features) treat as seq_len=1

        # x shape: (batch_size, seq_len, feature_dim)
        x = self.projection(x) # -> (batch_size, seq_len, transformer_dim)
        x = self.pos_encoder(x) # -> (batch_size, seq_len, transformer_dim)
        x = self.transformer_encoder(x) # -> (batch_size, seq_len, transformer_dim)

        # Global average pooling over sequence length
        x = torch.mean(x, dim=1) # -> (batch_size, transformer_dim)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNNTransformerModel_1(nn.Module):
    def __init__(self, feature_dim, num_classes=1, cnn_units=64, transformer_dim=64,
                 transformer_heads=4, transformer_layers=2, dropout_rate=0.2):
        super(CNNTransformerModel_p, self).__init__()
        self.feature_dim = feature_dim # Sequence length for CNN if input is (N,1,L)

        # CNN pathway
        self.conv1 = nn.Conv1d(1, cnn_units, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(cnn_units)
        self.conv2 = nn.Conv1d(cnn_units, cnn_units * 2, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(cnn_units * 2)

        self.use_pooling = feature_dim > 1
        cnn_current_len = feature_dim
        if self.use_pooling:
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            cnn_current_len = cnn_current_len // 2

        cnn_output_dim_flat = (cnn_units * 2) * cnn_current_len

        # Transformer pathway
        # For transformer, feature_dim is the embedding dimension of each element in sequence
        # If input is (Batch, Features), we treat it as (Batch, 1, Features) for transformer.
        # Or, if CNN processes (Batch, 1, SeqLen), its output could feed transformer.
        # Let's assume original feature_dim is input to transformer projection as well for parallel paths.
        self.transformer_projection = nn.Linear(feature_dim, transformer_dim) # Input is original features
        self.pos_encoder = PositionalEncoding(transformer_dim, dropout_rate)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4, dropout=dropout_rate, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=transformer_layers)

        # Combined pathway
        self.fc1 = nn.Linear(cnn_output_dim_flat + transformer_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        # Assume x input is (batch_size, feature_dim) or (batch_size, 1, feature_dim) for CNN
        # And (batch_size, feature_dim) for transformer (will be unsqueezed)

        if len(x.shape) == 2: # (batch_size, features)
            x_cnn_input = x.unsqueeze(1) # (batch_size, 1, features) for Conv1D
            x_trans_input = x.unsqueeze(1) # (batch_size, 1, features) for Transformer projection
        elif len(x.shape) == 3 and x.shape[1] == 1: # (batch_size, 1, features)
            x_cnn_input = x
            x_trans_input = x.squeeze(1) # (batch_size, features) for Linear projection
                                         # then unsqueeze for sequence.
            x_trans_input = x_trans_input.unsqueeze(1) # (batch_size, 1, features)
        else: # (batch_size, seq_len > 1, features)
            # This case needs clarification on how CNN and Transformer should process it.
            # Assuming feature_dim refers to the last dimension for transformer,
            # and the second to last for CNN (sequence length).
            # For simplicity, let's stick to the tabular-like input (N, F) or (N, 1, F)
            logger.error("CNNTransformerModel input shape not fully handled for (N, S>1, F). Assuming (N,F) or (N,1,F).")
            if len(x.shape) == 2:
                x_cnn_input = x.unsqueeze(1)
                x_trans_input = x.unsqueeze(1)
            else: # Fallback
                x_cnn_input = x
                x_trans_input = x.squeeze(1).unsqueeze(1) if x.shape[1] == 1 else x


        batch_size = x_cnn_input.size(0)

        # CNN pathway
        x_cnn = self.conv1(x_cnn_input)
        x_cnn = F.relu(x_cnn)
        x_cnn = self.batch_norm1(x_cnn)
        if self.use_pooling:
            x_cnn = self.pool(x_cnn)
        x_cnn = self.dropout(x_cnn)
        x_cnn = self.conv2(x_cnn)
        x_cnn = F.relu(x_cnn)
        x_cnn = self.batch_norm2(x_cnn)
        x_cnn = self.dropout(x_cnn)
        x_cnn = x_cnn.view(batch_size, -1) # Flatten CNN output

        # Transformer pathway
        # x_trans_input is (batch_size, 1, feature_dim)
        x_trans = self.transformer_projection(x_trans_input) # (batch, 1, transformer_dim)
        x_trans = self.pos_encoder(x_trans)      # (batch, 1, transformer_dim)
        x_trans = self.transformer_encoder(x_trans) # (batch, 1, transformer_dim)
        x_trans = torch.mean(x_trans, dim=1)     # (batch, transformer_dim) Global average pooling

        # Concatenate features
        x_combined = torch.cat((x_cnn, x_trans), dim=1)

        # Final layers
        x_combined = self.fc1(x_combined)
        x_combined = F.relu(x_combined)
        x_combined = self.dropout(x_combined)
        x_combined = self.fc2(x_combined)
        return x_combined


class CNNTransformerModel_2(nn.Module):
    def __init__(self, feature_dim, num_classes=1, cnn_units=64, transformer_dim=64,
                 transformer_heads=4, transformer_layers=2, dropout_rate=0.2,
                 # Add a flag to choose the architecture easily
                 architecture_mode='parallel'): # Options: 'parallel', 'sequential'
        
        super(CNNTransformerModel, self).__init__()
        self.feature_dim = feature_dim
        self.architecture_mode = architecture_mode
        self.dropout_layer = nn.Dropout(dropout_rate) # Shared dropout layer

        # --- CNN Pathway (Common to both modes) ---
        self.conv1 = nn.Conv1d(1, cnn_units, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(cnn_units)
        self.conv2 = nn.Conv1d(cnn_units, cnn_units * 2, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(cnn_units * 2)

        self.use_pooling = feature_dim > 1
        self.cnn_final_seq_len = feature_dim
        if self.use_pooling:
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            self.cnn_final_seq_len = self.cnn_final_seq_len // 2

        # CNN output channels = cnn_units * 2
        self.cnn_output_channels = cnn_units * 2
        # Flattened CNN output dimension
        cnn_output_dim_flat = self.cnn_output_channels * self.cnn_final_seq_len

        # --- Transformer Pathway Definition ---
        self.pos_encoder = PositionalEncoding(transformer_dim, dropout_rate)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4, dropout=dropout_rate, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=transformer_layers)

        # --- Mode-Specific Layers ---
        if self.architecture_mode == 'parallel':
            # Parallel mode: Transformer takes original features projected
            self.transformer_projection = nn.Linear(feature_dim, transformer_dim)
            # FC layer input combines flattened CNN and Transformer output
            combined_dim = cnn_output_dim_flat + transformer_dim
            
        elif self.architecture_mode == 'sequential':
            # Sequential mode: Transformer takes CNN output features
            # Input feature dimension for Transformer projection is # CNN output channels
            self.cnn_out_to_transformer_projection = nn.Linear(self.cnn_output_channels, transformer_dim)
            # FC layer input is just the Transformer output dimension
            combined_dim = transformer_dim
            
        else:
            raise ValueError(f"Unknown architecture_mode: {self.architecture_mode}")

        # --- Final Classification Head ---
        self.fc1 = nn.Linear(combined_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)


    def forward(self, x):
        # --- Input Shape Handling ---
        # Assume input x is (batch_size, feature_dim) or (batch_size, 1, feature_dim)
        if len(x.shape) == 2: # (batch_size, features) -> Need (batch_size, 1, features) for Conv1D
            x_cnn_input = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[1] == 1: # (batch_size, 1, features)
            x_cnn_input = x
        else:
            # Handle other cases or raise error if shape is unexpected
            # Using your existing fallback for now:
            logger.error("CNNTransformerModel input shape unexpected. Assuming (N, F) or (N, 1, F).")
            if len(x.shape) == 2:
                x_cnn_input = x.unsqueeze(1)
            else: # Fallback for (N, S>1, F) or other shapes
                x_cnn_input = x if x.shape[1] == 1 else x.unsqueeze(1) # Ensure C=1 if possible


        batch_size = x_cnn_input.size(0)

        # --- CNN Pathway Execution (Common) ---
        # Output shape: (batch_size, self.cnn_output_channels, self.cnn_final_seq_len)
        x_cnn_out = self.conv1(x_cnn_input)
        x_cnn_out = F.relu(x_cnn_out)
        x_cnn_out = self.batch_norm1(x_cnn_out)
        if self.use_pooling:
            x_cnn_out = self.pool(x_cnn_out)
        x_cnn_out = self.dropout_layer(x_cnn_out) # Apply dropout

        x_cnn_out = self.conv2(x_cnn_out)
        x_cnn_out = F.relu(x_cnn_out)
        x_cnn_out = self.batch_norm2(x_cnn_out)
        x_cnn_out = self.dropout_layer(x_cnn_out) # Apply dropout


        # --- Parallel Architecture ---
        if self.architecture_mode == 'parallel':
            # 1. Prepare Transformer Input (from original x)
            if len(x.shape) == 3 and x.shape[1] == 1: # If input was (N, 1, F)
                x_trans_input_orig = x.squeeze(1) # (N, F) for projection
            else: # Assume input was (N, F)
                x_trans_input_orig = x
            x_trans_input_orig = x_trans_input_orig.unsqueeze(1) # (N, 1, F) - Seq len 1

            # 2. Transformer Pathway Execution
            x_trans = self.transformer_projection(x_trans_input_orig) # (batch, 1, transformer_dim)
            x_trans = self.pos_encoder(x_trans)
            x_trans = self.transformer_encoder(x_trans) # (batch, 1, transformer_dim)
            x_trans_pooled = torch.mean(x_trans, dim=1) # (batch, transformer_dim) Global average pooling

            # 3. Flatten CNN Output
            x_cnn_flat = x_cnn_out.view(batch_size, -1)

            # 4. Concatenate features
            x_combined = torch.cat((x_cnn_flat, x_trans_pooled), dim=1)


        # --- Sequential Architecture (CNN -> Transformer) ---
        elif self.architecture_mode == 'sequential':
            # 1. Prepare Transformer Input (from CNN output)
            # Reshape CNN output: (batch, channels, seq_len) -> (batch, seq_len, channels)
            x_trans_input_seq = x_cnn_out.permute(0, 2, 1) # (batch, cnn_final_seq_len, cnn_output_channels)

            # 2. Project features from CNN channels to transformer_dim
            x_trans_input_seq = self.cnn_out_to_transformer_projection(x_trans_input_seq) # (batch, cnn_final_seq_len, transformer_dim)

            # 3. Transformer Pathway Execution
            x_trans = self.pos_encoder(x_trans_input_seq)
            x_trans = self.transformer_encoder(x_trans) # (batch, cnn_final_seq_len, transformer_dim)
            
            # 4. Pool Transformer Output (Global Average Pooling over the sequence length)
            x_combined = torch.mean(x_trans, dim=1) # (batch, transformer_dim)


        # --- Final Classification Head (Common) ---
        x_combined = self.fc1(x_combined)
        x_combined = F.relu(x_combined)
        x_combined = self.dropout_layer(x_combined) # Apply dropout
        x_combined = self.fc2(x_combined)

        return x_combined

class CNNTransformerModel(nn.Module):
    def __init__(self, feature_dim, num_classes=1, 
                 cnn_units=32, # Shallower default
                 transformer_dim=32, # Shallower default
                 transformer_heads=2,  # Shallower default
                 transformer_layers=1, # Shallower default (significant reduction)
                 architecture_mode='parallel', 
                 dropout_rate=0.3, # Default general dropout, can be slightly higher for small datasets
                 cnn_dropout=None,      
                 transformer_dropout=None, 
                 fc_dropout=None):      
        
        super(CNNTransformerModel, self).__init__()
        self.feature_dim = feature_dim
        self.architecture_mode = architecture_mode
        
        # --- Explicitly cast to integer types ---
        _cnn_units = int(cnn_units)
        transformer_dim = int(transformer_dim)
        transformer_heads = int(transformer_heads)
        transformer_layers = int(transformer_layers)
        # -----

        _cnn_dropout_rate = cnn_dropout if cnn_dropout is not None else dropout_rate
        _transformer_dropout_rate = transformer_dropout if transformer_dropout is not None else dropout_rate
        _fc_dropout_rate = fc_dropout if fc_dropout is not None else dropout_rate

        # Ensure transformer_dim is divisible by transformer_heads
        if transformer_dim % transformer_heads != 0:
            # Adjust dim or heads, or raise error. For simplicity, let's adjust dim.
            original_transformer_dim = transformer_dim
            transformer_dim = (transformer_dim // transformer_heads) * transformer_heads
            if transformer_dim == 0 and original_transformer_dim > 0: # if dim < heads
                transformer_dim = transformer_heads 
            logger.warning(f"CNNTransformerModel: transformer_dim ({original_transformer_dim}) not divisible by "
                           f"heads ({transformer_heads}). Adjusted transformer_dim to {transformer_dim}.")
            if transformer_dim == 0 and original_transformer_dim > 0:
                 raise ValueError(f"Cannot run transformer with original_dim {original_transformer_dim} and heads {transformer_heads}. Adjusted dim is 0.")


        logger.info(f"CNNTransformer Init: cnn_units={cnn_units}, transformer_dim={transformer_dim}, "
                    f"transformer_heads={transformer_heads}, transformer_layers={transformer_layers}, "
                    f"cnn_dropout={_cnn_dropout_rate}, transformer_dropout={_transformer_dropout_rate}, "
                    f"fc_dropout={_fc_dropout_rate}, arch_mode={architecture_mode}")

        # --- Dropout Layers ---
        self.cnn_dropout_layer = nn.Dropout(_cnn_dropout_rate)
        self.fc_dropout_layer = nn.Dropout(_fc_dropout_rate)

        # --- CNN Pathway ---
        _cnn_units_int = int(cnn_units[0]) if isinstance(cnn_units, (list, tuple)) else int(cnn_units)
        
        self.conv1 = nn.Conv1d(1, _cnn_units_int, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(_cnn_units_int)
        # Second CNN layer with same number of units to keep it shallower
        self.conv2 = nn.Conv1d(_cnn_units_int, _cnn_units_int, kernel_size=3, padding=1) 
        self.batch_norm2 = nn.BatchNorm1d(_cnn_units_int)

        self.use_pooling = self.feature_dim > 4 # Only pool if sequence length is somewhat substantial after convs
        self.cnn_final_seq_len = self.feature_dim 
        if self.use_pooling:
            self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            self.cnn_final_seq_len = self.cnn_final_seq_len // 2

        self.cnn_output_channels = _cnn_units_int # Adjusted due to conv2 change
        cnn_output_dim_flat = self.cnn_output_channels * self.cnn_final_seq_len

        # --- Transformer Pathway Definition ---
        self.pos_encoder = PositionalEncoding(transformer_dim, dropout=_transformer_dropout_rate)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=transformer_dim, nhead=transformer_heads,
            dim_feedforward=transformer_dim * 2, # Reduced feedforward dim as well (common: 2*d_model to 4*d_model)
            dropout=_transformer_dropout_rate, 
            activation=F.gelu, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=transformer_layers)

        # --- Mode-Specific Layers & Fusion ---
        if self.architecture_mode == 'parallel':
            self.transformer_projection = nn.Linear(feature_dim, transformer_dim)
            self.parallel_fusion_dim = cnn_output_dim_flat + transformer_dim
            # Simpler fusion: just one linear layer, or make it smaller
            fused_intermediate_dim = max(32, self.parallel_fusion_dim // 4) # Ensure it's not too small
            self.parallel_fusion_layer = nn.Sequential(
                nn.Linear(self.parallel_fusion_dim, fused_intermediate_dim),
                nn.ReLU(), 
                nn.Dropout(_fc_dropout_rate) 
            )
            combined_dim_for_fc = fused_intermediate_dim
            
        elif self.architecture_mode == 'sequential':
            self.cnn_out_to_transformer_projection = nn.Linear(self.cnn_output_channels, transformer_dim)
            combined_dim_for_fc = transformer_dim
        else:
            raise ValueError(f"Unknown architecture_mode: {self.architecture_mode}")

        # --- Final Classification Head ---
        fc1_hidden_dim = max(32, combined_dim_for_fc // 2) # Make FC1 adaptable and not too large
        self.fc1 = nn.Linear(combined_dim_for_fc, fc1_hidden_dim)
        self.fc2 = nn.Linear(fc1_hidden_dim, num_classes)

    def forward(self, x):
        original_input_x = x.clone() # Keep original x for parallel transformer path if needed

        if len(x.shape) == 2: 
            x_cnn_input = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[1] == 1: 
            x_cnn_input = x
        else: 
            logger.warning(f"CNNTransformerModel input shape {x.shape} unexpected. Attempting to reshape for CNN.")
            if len(x.shape) > 2 and x.shape[-1] == self.feature_dim: 
                x_cnn_input = x.mean(dim=1, keepdim=True) 
                logger.warning(f"Input reshaped from {x.shape} to {x_cnn_input.shape} for CNN by averaging over dim 1.")
            elif len(x.shape) == 2:
                 x_cnn_input = x.unsqueeze(1)
            else: 
                 x_cnn_input = x[:,0,:].unsqueeze(1) if x.shape[0]>0 and x.shape[1]>0 else x # Fallback, take first "channel"

        batch_size = x_cnn_input.size(0)

        # --- CNN Pathway Execution ---
        x_cnn_out = self.conv1(x_cnn_input)
        x_cnn_out = F.relu(self.batch_norm1(x_cnn_out))
        if self.use_pooling:
            x_cnn_out = self.pool(x_cnn_out)
        x_cnn_out = self.cnn_dropout_layer(x_cnn_out)

        x_cnn_out = self.conv2(x_cnn_out)
        x_cnn_out = F.relu(self.batch_norm2(x_cnn_out))
        x_cnn_out = self.cnn_dropout_layer(x_cnn_out) 

        # --- Mode-Specific Processing ---
        if self.architecture_mode == 'parallel':
            x_for_transformer_projection = original_input_x.squeeze(1) if len(original_input_x.shape) == 3 and original_input_x.shape[1] == 1 else original_input_x
            if len(x_for_transformer_projection.shape) == 2: # Ensure (N, S, E) for transformer
                x_trans_input_orig_seq_len_1 = x_for_transformer_projection.unsqueeze(1)
            else: # If it's already (N,S,E) but S might not be 1
                # This case needs careful thought: if original input is (N,S,F) and S > 1,
                # the current self.transformer_projection(Linear(F, transformer_dim)) might not be ideal
                # if you want to process the sequence.
                # For S=1, (N,1,F) is fine.
                # If S > 1, you might project each of S elements.
                # For now, assuming S=1 if input was (N,F) or (N,1,F).
                if x_for_transformer_projection.shape[1] != 1:
                    logger.warning(f"Parallel transformer path received input with seq_len > 1 ({x_for_transformer_projection.shape}). Taking mean over seq_len for projection.")
                    x_trans_input_orig_seq_len_1 = x_for_transformer_projection.mean(dim=1, keepdim=True) # (N, 1, F)
                else:
                    x_trans_input_orig_seq_len_1 = x_for_transformer_projection


            x_trans = self.transformer_projection(x_trans_input_orig_seq_len_1) 
            x_trans = self.pos_encoder(x_trans)
            x_trans = self.transformer_encoder(x_trans) 
            x_trans_pooled = torch.mean(x_trans, dim=1) 

            x_cnn_flat = x_cnn_out.view(batch_size, -1)
            x_concatenated = torch.cat((x_cnn_flat, x_trans_pooled), dim=1)
            x_fused = self.parallel_fusion_layer(x_concatenated) 
            x_to_fc = x_fused

        elif self.architecture_mode == 'sequential':
            x_trans_input_seq = x_cnn_out.permute(0, 2, 1) 
            x_trans_input_seq = self.cnn_out_to_transformer_projection(x_trans_input_seq) 
            x_trans = self.pos_encoder(x_trans_input_seq)
            x_trans = self.transformer_encoder(x_trans) 
            x_to_fc = torch.mean(x_trans, dim=1) 

        # --- Final Classification Head ---
        x_out = self.fc1(x_to_fc)
        x_out = F.relu(x_out)
        x_out = self.fc_dropout_layer(x_out)
        x_out = self.fc2(x_out)

        return x_out


# --- 2. TRAINING AND EVALUATION FUNCTIONS ---
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)


    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model to {self.path} ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def reset_parameters(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()



def train_model_with_cv(model_class, model_params, X_train_all, y_train_all,
                      optimizer_params, criterion, device,
                      model_name_base, report_dir, subdirs,
                      is_binary, num_class, 
                      n_splits=5, epochs=30, batch_size=16, has_channel_dim_input=True):
    
    """Train model with cross-validation, save best fold model, and train final model."""
    # X_train_all, y_train_all are full training set (before splitting into CV folds)
    # These should be tensors already on the correct device potentially.

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics_list = []
    best_fold_f1 = -1
    best_fold_model_path = None
    best_fold_num = -1

    # Convert to numpy for skf.split if they are tensors
    X_train_all_np = X_train_all.cpu().numpy() if isinstance(X_train_all, torch.Tensor) else X_train_all
    y_train_all_np = y_train_all.cpu().numpy() if isinstance(y_train_all, torch.Tensor) else y_train_all


    # Handle original data shape for CNN input
    # Input to models (CNN, Transformer, CNNTransformer) expects (N, C, L) or (N, L) or (N, S, F)
    # Let's assume data_dict['X_train'] is (N, C, L) if has_channel_dim_input is true, else (N, L)
    # where L is feature_dim.

    logger.info(f"Starting {n_splits}-fold cross-validation for {model_name_base}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_all_np, y_train_all_np)):
        logger.info(f"--- Fold {fold + 1}/{n_splits} ---")
        model_name_fold = f"{model_name_base}_fold{fold + 1}"
        fold_model_save_path = subdirs['models'] / f"{model_name_fold}_best.pt"

        model = model_class(**model_params).to(device)
        model.apply(reset_parameters) # Re-initialize weights for each fold

        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=False)
        early_stopping = EarlyStopping(patience=10, verbose=False, path=fold_model_save_path)

        # _X_train_fold = torch.tensor(X_train_all_np[train_idx], dtype=torch.float32)
        # _y_train_fold = torch.tensor(y_train_all_np[train_idx], dtype=torch.float32).unsqueeze(1)
        # _X_val_fold = torch.tensor(X_train_all_np[val_idx], dtype=torch.float32)
        # _y_val_fold = torch.tensor(y_train_all_np[val_idx], dtype=torch.float32).unsqueeze(1)
        # Target tensor preparation for CV fold
        target_dtype_fold = torch.long if not is_binary and num_class > 1 else torch.float32
        
        _X_train_fold = torch.tensor(X_train_all_np[train_idx], dtype=torch.float32)
        _y_train_fold_np = y_train_all_np[train_idx]
        _y_train_fold = torch.tensor(_y_train_fold_np, dtype=target_dtype_fold)
        
        _X_val_fold = torch.tensor(X_train_all_np[val_idx], dtype=torch.float32)
        _y_val_fold_np = y_train_all_np[val_idx]
        _y_val_fold = torch.tensor(_y_val_fold_np, dtype=target_dtype_fold)

        # Reshape y only if binary classification for BCEWithLogitsLoss
        if is_binary: # Binary mode
            if len(_y_train_fold.shape) == 1: _y_train_fold = _y_train_fold.unsqueeze(1)
            if len(_y_val_fold.shape) == 1: _y_val_fold = _y_val_fold.unsqueeze(1)

        if has_channel_dim_input:
            if len(_X_train_fold.shape) == 2: _X_train_fold = _X_train_fold.unsqueeze(1)
            if len(_X_val_fold.shape) == 2: _X_val_fold = _X_val_fold.unsqueeze(1)

        train_dataset_fold = TensorDataset(_X_train_fold.to(device), _y_train_fold.to(device))
        val_dataset_fold = TensorDataset(_X_val_fold.to(device), _y_val_fold.to(device))
        train_loader_fold = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True)
        val_loader_fold = DataLoader(val_dataset_fold, batch_size=batch_size, shuffle=False)

        fold_train_losses, fold_val_losses = [], []

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader_fold:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                running_loss += loss.item()
            epoch_train_loss = running_loss / len(train_loader_fold)
            fold_train_losses.append(epoch_train_loss)

            model.eval()
            val_loss = 0.0
            # current_val_preds, current_val_labels = [], []
            current_val_preds_raw_outputs = [] # Store raw model outputs if needed for multiclass AUC
            current_val_preds_indices = [] # Store predicted class indices
            current_val_labels_true = []
            with torch.no_grad():
                for inputs, labels in val_loader_fold:
                    inputs, ground_truth_labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs) # Shape: [batch, 1] for binary, [batch, num_classes] for multiclass
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    # probs = torch.sigmoid(outputs)
                    # preds = (probs >= 0.5).float()
                    # current_val_preds.extend(preds.cpu().numpy())
                    # current_val_labels.extend(labels.cpu().numpy())
                    if not is_binary and num_class > 1: # Multiclass
                        probs = torch.softmax(outputs, dim=1)
                        preds_indices = torch.argmax(probs, dim=1)
                        current_val_preds_raw_outputs.extend(probs.cpu().numpy()) # Store all class probabilities
                        current_val_preds_indices.extend(preds_indices.cpu().numpy())
                    else: # Binary
                        # For BCEWithLogitsLoss, labels_in_batch is [batch,1], outputs is [batch,1]
                        probs = torch.sigmoid(outputs)
                        preds_indices = (probs >= 0.5).float().squeeze(-1) # Squeeze to make it 1D for metrics if needed
                        current_val_preds_raw_outputs.extend(probs.cpu().numpy()) # Store sigmoid probabilities
                        current_val_preds_indices.extend(preds_indices.cpu().numpy())
                    current_val_labels_true.extend(ground_truth_labels.cpu().numpy().squeeze()) # Squeeze in case labels were [N,1] for binary


            epoch_val_loss = val_loss / len(val_loader_fold)
            fold_val_losses.append(epoch_val_loss)
            scheduler.step(epoch_val_loss)

            # y_true_val_np = np.array(current_val_labels).flatten()
            # y_pred_val_np = np.array(current_val_preds).flatten()
            # f1_val = f1_score(y_true_val_np, y_pred_val_np, average='binary', zero_division=0) # binary
            # f1_val = f1_score(y_true_val_np, y_pred_val_np, average='macro', zero_division=0) # multiclass
            y_true_val_np = np.array(current_val_labels_true).flatten() # Ensure 1D for sklearn metrics
            y_pred_val_np = np.array(current_val_preds_indices).flatten() # Ensure 1D for sklearn metrics
            y_prob_val_np = np.array(current_val_preds_raw_outputs) # Shape [N, num_classes] or [N, 1] or [N]

            avg_setting_multiclass = 'macro' # Or 'weighted'

            # Metrics calculation
            accuracy_val = accuracy_score(y_true_val_np, y_pred_val_np)
            if not is_binary and num_class > 1: # Multiclass
                f1_val = f1_score(y_true_val_np, y_pred_val_np, average=avg_setting_multiclass, zero_division=0)
                precision_val, recall_val, _, _ = precision_recall_fscore_support(y_true_val_np, y_pred_val_np, average=avg_setting_multiclass, zero_division=0)
            else: # Binary
                f1_val = f1_score(y_true_val_np, y_pred_val_np, average='binary', zero_division=0)
                precision_val, recall_val, _, _ = precision_recall_fscore_support(y_true_val_np, y_pred_val_np, average='binary', zero_division=0)
          

            logger.debug(f"Epoch {epoch+1}/{epochs}, Fold {fold+1} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val F1: {f1_val:.4f}, precision_val: {precision_val:.4f}, recall_val: {recall_val:.4f}")
            early_stopping(epoch_val_loss, model)
            if early_stopping.early_stop:
                logger.info(f"Early stopping at epoch {epoch + 1} for fold {fold + 1}")
                break

        # Load best model for this fold for evaluation
        model.load_state_dict(torch.load(fold_model_save_path))
        model.eval()
        
        # val_preds_final, val_labels_final, val_probs_final = [], [], []
        # with torch.no_grad():
        #     for inputs, labels in val_loader_fold:
        #         outputs = model(inputs)
        #         probs = torch.sigmoid(outputs)
        #         preds = (probs >= 0.5).float()
        #         val_probs_final.extend(probs.cpu().numpy())
        #         val_preds_final.extend(preds.cpu().numpy())
        #         val_labels_final.extend(labels.cpu().numpy())
        
         # Re-initialize lists for final evaluation of this fold's best model
        val_preds_final_indices_list = [] # Store predicted class indices
        val_labels_final_true_list = []   # Store true labels
        val_probs_final_raw_list = []     # Store probabilities
        with torch.no_grad():
            for inputs, labels_in_batch in val_loader_fold: # labels_in_batch from loader
                inputs_dev = inputs.to(device)
                ground_truth_labels_dev = labels_in_batch.to(device) # These are the true labels
                
                outputs = model(inputs_dev) # Raw logits from the model

                # Use 'is_binary' and 'num_class' as defined in your train_model_with_cv signature
                if not is_binary and num_class > 1: # Multiclass
                    probs = torch.softmax(outputs, dim=1) # Shape [B, num_class]
                    preds_indices = torch.argmax(probs, dim=1) # Shape [B]
                    
                    val_probs_final_raw_list.extend(probs.cpu().numpy().tolist()) # Store list of lists/arrays
                    val_preds_final_indices_list.extend(preds_indices.cpu().numpy().tolist())
                    # ground_truth_labels_dev for multiclass (CrossEntropy) is 1D [B] long
                    val_labels_final_true_list.extend(ground_truth_labels_dev.cpu().numpy().tolist())
                else: # Binary
                    probs = torch.sigmoid(outputs) # Shape [B, 1]
                    preds_indices = (probs >= 0.5).float().squeeze(-1) # Squeeze to [B]
                    
                    val_probs_final_raw_list.extend(probs.cpu().numpy().tolist())
                    val_preds_final_indices_list.extend(preds_indices.cpu().numpy().tolist())
                    # ground_truth_labels_dev for binary (BCEWithLogits) is typically [B, 1] float
                    val_labels_final_true_list.extend(ground_truth_labels_dev.cpu().numpy().squeeze(-1).tolist())


        # y_true_np = np.array(val_labels_final).flatten()
        # y_pred_np = np.array(val_preds_final).flatten()
        # y_prob_np = np.array(val_probs_final).flatten()
        # Now create y_true_np and y_pred_np from these correct lists
        y_true_np = np.array(val_labels_final_true_list).flatten() 
        y_pred_np = np.array(val_preds_final_indices_list).flatten()
        # For y_prob_np, it needs careful handling if it was a list of lists/arrays
        if val_probs_final_raw_list:
            if isinstance(val_probs_final_raw_list[0], list) or \
               (isinstance(val_probs_final_raw_list[0], np.ndarray) and val_probs_final_raw_list[0].ndim > 0 and val_probs_final_raw_list[0].shape[0] > 1) : # Check if it's list of lists/arrays for multiclass probs
                try: # Stacking list of [B, num_class] arrays (multiclass) or [B,1] arrays (binary)
                    y_prob_np = np.array([item for sublist in val_probs_final_raw_list for item in (sublist if isinstance(sublist, list) else [sublist])])
                    if not is_binary and num_class > 1 and y_prob_np.ndim == 1 and len(val_probs_final_raw_list) > 0: # Check if y_prob_np needs reshaping for multiclass
                        num_samples_local = len(y_true_np)
                        if y_prob_np.shape[0] == num_samples_local * num_class :
                             y_prob_np = y_prob_np.reshape(num_samples_local, num_class)
                    elif is_binary and y_prob_np.ndim == 1 and len(val_probs_final_raw_list) > 0:
                         y_prob_np = y_prob_np.reshape(-1,1) # Ensure it's [N,1] for consistency if it became flat
                except: # Fallback if vstacking complex lists of lists fails
                    logger.warning("Could not robustly stack probabilities for y_prob_np. Using simple np.array.")
                    y_prob_np = np.array(val_probs_final_raw_list) # This might need further processing based on exact content
            else: # Likely list of scalars (binary probabilities already squeezed)
                y_prob_np = np.array(val_probs_final_raw_list)
        else:
            y_prob_np = np.array([])

        fold_accuracy = accuracy_score(y_true_np, y_pred_np)
        # fold_precision, fold_recall, fold_f1, _ = precision_recall_fscore_support(y_true_np, y_pred_np, average='binary', zero_division=0)
        # fold_roc_auc = 0
        # if len(np.unique(y_true_np)) > 1: # AUC requires more than 1 class in y_true
        #     try:
        #         fold_fpr, fold_tpr, _ = roc_curve(y_true_np, y_prob_np)
        #         fold_roc_auc = auc(fold_fpr, fold_tpr)
        #     except Exception as e_auc:
        #         logger.warning(f"Could not calculate AUC for fold {fold+1}: {e_auc}")
        #         fold_fpr, fold_tpr = np.array([]), np.array([])
        # else:
        #     fold_fpr, fold_tpr = np.array([]), np.array([])
        
        if not is_binary and num_class > 1: # Multiclass
            fold_precision, fold_recall, fold_f1, _ = precision_recall_fscore_support(y_true_np, y_pred_np, average=avg_setting_multiclass, zero_division=0)
            fold_roc_auc = 0
            if len(np.unique(y_true_np)) > 1: # Check if more than one class truly present in this fold's y_true
                try:
                    # y_prob_np should be [n_samples, n_classes] from softmax
                    fold_roc_auc = roc_auc_score(y_true_np, y_prob_np, multi_class='ovr', average=avg_setting_multiclass)
                except Exception as e_auc:
                    logger.warning(f"Could not calculate multiclass AUC for fold {fold+1}: {e_auc}")
            fold_fpr, fold_tpr = None, None # roc_curve is for binary; skip for multiclass or do per-class
        else: # Binary
            fold_precision, fold_recall, fold_f1, _ = precision_recall_fscore_support(y_true_np, y_pred_np, average='binary', zero_division=0)
            fold_roc_auc = 0
            if len(np.unique(y_true_np)) > 1:
                try:
                    # y_prob_np here is from sigmoid, shape [N] or [N,1]
                    binary_y_prob = y_prob_np.squeeze() if y_prob_np.ndim > 1 else y_prob_np
                    fold_fpr, fold_tpr, _ = roc_curve(y_true_np, binary_y_prob)
                    fold_roc_auc = auc(fold_fpr, fold_tpr)
                except Exception as e_auc:
                    logger.warning(f"Could not calculate binary AUC for fold {fold+1}: {e_auc}")
                    fold_fpr, fold_tpr = np.array([]), np.array([])
            else:
                fold_fpr, fold_tpr = np.array([]), np.array([])


        fold_metrics = {
            'fold': fold + 1, 'accuracy': fold_accuracy, 'precision': fold_precision,
            'recall': fold_recall, 'f1': fold_f1, 'auc': fold_roc_auc,
            'val_loss': early_stopping.val_loss_min,
            'train_losses': fold_train_losses, 'val_losses': fold_val_losses,
            'fpr': (fold_fpr.tolist() if fold_fpr is not None else []),
            'tpr': (fold_tpr.tolist() if fold_tpr is not None else []),
            'y_true': y_true_np.tolist(), 'y_pred': y_pred_np.tolist(), 'y_prob': y_prob_np.tolist(),
            'confusion_matrix': confusion_matrix(y_true_np, y_pred_np).tolist()
        }
        fold_metrics_list.append(fold_metrics)
        save_model_metrics(fold_metrics, report_dir, model_name_fold, subfolder=subdirs['metrics'].name) # Pass subdirs['metrics'] directly or its name

        fig_train_curve = plot_training_curves(fold_train_losses, fold_val_losses, model_name_fold)
        save_figure(fig_train_curve, f"{model_name_fold}_training_curves", subdirs['figures'])

        if fold_f1 > best_fold_f1:
            best_fold_f1 = fold_f1
            best_fold_model_path = fold_model_save_path
            best_fold_num = fold + 1

    # --- After all folds ---
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in fold_metrics_list]),
        'precision': np.mean([m['precision'] for m in fold_metrics_list]),
        'recall': np.mean([m['recall'] for m in fold_metrics_list]),
        'f1': np.mean([m['f1'] for m in fold_metrics_list]),
        'auc': np.mean([m['auc'] for m in fold_metrics_list if m['auc'] is not None]), # handle None AUC
        'std_f1': np.std([m['f1'] for m in fold_metrics_list]),
    }
    # Calculate F1 CI for the combined validation sets
    all_folds_y_true = np.concatenate([m['y_true'] for m in fold_metrics_list])
    all_folds_y_pred = np.concatenate([m['y_pred'] for m in fold_metrics_list])    
    val_f1 = f1_score(all_folds_y_true, all_folds_y_pred, average='binary', zero_division=0)
    _, val_f1_lower, val_f1_upper = calculate_f1_ci(all_folds_y_true, all_folds_y_pred)
    avg_metrics['f1_score_val'] = val_f1
    avg_metrics['f1_95_ci_val'] = f"[{val_f1_lower:.4f}, {val_f1_upper:.4f}]"
    
    
    logger.info(f"Avg CV F1 for {model_name_base}: {avg_metrics['f1']:.4f} +/- {avg_metrics['std_f1']:.4f}")
    with open(subdirs['metrics'] / f"{model_name_base}_cv_summary.json", 'w') as f:
        json.dump({'avg_metrics': avg_metrics, 'fold_metrics': fold_metrics_list}, f, indent=4)
        
    # --- Determine epochs for final model training ---
    epochs_for_final_training = epochs  # Default to the original 'epochs' argument 
    if best_fold_num != -1 and fold_metrics_list:
        try:
            # best_fold_num is 1-indexed
            best_fold_metrics_data = next(item for item in fold_metrics_list if item["fold"] == best_fold_num)
            num_epochs_in_best_fold = len(best_fold_metrics_data.get('train_losses', []))
            
            if num_epochs_in_best_fold > 0:
                epochs_for_final_training = num_epochs_in_best_fold
                logger.info(f"Best CV fold ({best_fold_num}) completed {num_epochs_in_best_fold} epochs. "
                            f"Using this epoch count for final model training.")
            else:
                logger.warning(f"Could not determine epoch count from best CV fold ({best_fold_num}) as 'train_losses' was empty or missing. "
                               f"Defaulting to {epochs} epochs for final model training.")
        except StopIteration:
            logger.warning(f"Metrics for the best CV fold ({best_fold_num}) not found in fold_metrics_list. "
                           f"Defaulting to {epochs} epochs for final model training.")
    else:
        logger.warning(f"Best CV fold was not identified or fold_metrics_list is empty. "
                       f"Defaulting to {epochs} epochs for final model training.")
    
    

    # --- Train final model on all X_train_all, y_train_all ---
    logger.info(f"Training final model ({model_name_base}) on all training data using best fold ({best_fold_num}) epoch count or fixed epochs.")
    final_model_name = f"{model_name_base}_final"
    final_model_save_path = subdirs['models'] / f"{final_model_name}_best.pt"

    final_model = model_class(**model_params).to(device)
    final_model.apply(reset_parameters)
    # Optionally, load weights from the best CV fold model as a starting point, or just re-train
    if best_fold_model_path and os.path.exists(best_fold_model_path) and False: # Disabled for now, retrain from scratch
        logger.info(f"Loading weights from best fold model: {best_fold_model_path}")
        final_model.load_state_dict(torch.load(best_fold_model_path))

    optimizer_final = torch.optim.AdamW(final_model.parameters(), **optimizer_params)
    scheduler_final = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_final, 'min', patience=5, factor=0.5, verbose=False)
    # For final model, we usually train for a fixed number of epochs or use early stopping against a validation set (if available)
    # Here, we'll use early stopping against the *test* set provided to the main caller (or a dedicated validation set if split earlier)
    # For now, let's train for a fixed number of epochs, or use epochs from best fold.
    # Or, more simply, train for the specified 'epochs' argument, using early stopping based on a small part of itself if no val set for final training
    # Let's assume X_train_all, y_train_all is used entirely for training the final model. Evaluation will be on a separate test set.

    # Prepare X_train_all_tensor for the final model
    _X_train_all_tensor = X_train_all.to(device) if isinstance(X_train_all, torch.Tensor) else torch.tensor(X_train_all_np, dtype=torch.float32).to(device)
    # _y_train_all_tensor = y_train_all.to(device) if isinstance(y_train_all, torch.Tensor) else torch.tensor(y_train_all_np, dtype=torch.float32).unsqueeze(1).to(device)

    if has_channel_dim_input: # Ensure channel dim for CNN-like models
        if len(_X_train_all_tensor.shape) == 2: _X_train_all_tensor = _X_train_all_tensor.unsqueeze(1)

    # --- edit ---    
    # Ensure correct type and device, then reshape if necessary
    # _y_tensor_for_final_model = y_train_all.clone().detach().to(dtype=torch.float32, device=device)
    # if len(_y_tensor_for_final_model.shape) == 1:
    #     _y_tensor_for_final_model = _y_tensor_for_final_model.unsqueeze(1) # Make it [N, 1]
        
    target_dtype_final = torch.long if not is_binary and num_class > 1 else torch.float32
    _y_tensor_for_final_model = y_train_all.clone().detach().to(dtype=target_dtype_final, device=device)
    if is_binary: # Binary mode
        if len(_y_tensor_for_final_model.shape) == 1: _y_tensor_for_final_model = _y_tensor_for_final_model.unsqueeze(1)

    # --- edit ---

    final_train_dataset = TensorDataset(_X_train_all_tensor, _y_tensor_for_final_model)
    final_train_loader = DataLoader(final_train_dataset, batch_size=batch_size, shuffle=True)

    # We need a validation set for early stopping the final model.
    # If the user doesn't provide X_test, y_test to *this CV function*, we can't use early stopping with external val data.
    # For now, let's assume the `epochs` parameter is well-chosen, or we train for a bit longer.
    # A better approach for the "final" model is to train it and then evaluate on the holdout set.
    # The early stopping here would be on a split of X_train_all if no separate validation set is passed.
    # Let's train for `epochs` and save the last model, or best if we had a val set.
    # For simplicity, save the model at the end of training. The CV selected the architecture/hyperparams.

    final_train_losses = []
    # logger.info(f"Training final model {final_model_name} for {epochs} epochs.") 
    # adjust: epochs_for_final_training
    logger.info(f"Training final model {final_model_name} for {epochs_for_final_training} epochs.") 
    
    for epoch in range(epochs_for_final_training): # Using epochs_for_final_training instead of epochs
        final_model.train()
        running_loss = 0.0
        for inputs, labels in final_train_loader:
            optimizer_final.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            optimizer_final.step()
            running_loss += loss.item()
        epoch_train_loss = running_loss / len(final_train_loader)
        final_train_losses.append(epoch_train_loss)
        # No scheduler step here if no validation loss. Could use dummy.
        if epoch % 10 == 0 or epoch == epochs -1 :
            logger.info(f"Final Model Training - Epoch {epoch+1}/{epochs_for_final_training}, Train Loss: {epoch_train_loss:.4f}") #Epoch {epoch+1}/{epochs}

    torch.save(final_model.state_dict(), final_model_save_path)
    logger.info(f"Saved final trained model to {final_model_save_path}")

    # Evaluation of this final_model on X_test will happen outside this CV function, in the main script.
    # This function returns the path to the *best fold* model and the *final trained* model.
    return final_model_save_path, best_fold_model_path, avg_metrics

def evaluate_model(model, test_loader, criterion, device,
                   model_name="Model", report_dir=None, subdirs=None,
                   is_binary=True, num_classes=2, class_labels_for_plots=None):
    model.eval()
    test_loss = 0.0
    
    # Store lists of NumPy arrays (one array per batch)
    batch_labels_list = []
    batch_preds_list = []
    batch_probs_list = []

    with torch.no_grad():
        for inputs, labels_in_batch in test_loader:
            inputs_dev = inputs.to(device)
            ground_truth_labels_dev = labels_in_batch.to(device)
            outputs = model(inputs_dev) # Raw logits
            
            loss = criterion(outputs, ground_truth_labels_dev)
            test_loss += loss.item()

            # --- Predictions and Probabilities ---
            current_batch_preds_np = None
            current_batch_probs_np = None
            current_batch_labels_np = None

            if not is_binary and num_classes > 1: # Multiclass
                probs = torch.softmax(outputs, dim=1) # Shape [B, num_classes]
                preds_indices = torch.argmax(probs, dim=1) # Shape [B]
                
                current_batch_probs_np = probs.cpu().numpy()
                current_batch_preds_np = preds_indices.cpu().numpy()
                current_batch_labels_np = ground_truth_labels_dev.cpu().numpy() # Shape [B] (long)
            else: # Binary
                probs = torch.sigmoid(outputs) # Shape [B, 1]
                preds_indices = (probs >= 0.5).float().squeeze(-1) # Shape [B] (float 0.0 or 1.0)
                
                current_batch_probs_np = probs.cpu().numpy() # Shape [B, 1]
                current_batch_preds_np = preds_indices.cpu().numpy()
                # ground_truth_labels_dev is [B, 1] Float for BCEWithLogitsLoss
                current_batch_labels_np = ground_truth_labels_dev.cpu().numpy().squeeze(-1) # Shape [B]

            batch_labels_list.append(current_batch_labels_np)
            batch_preds_list.append(current_batch_preds_np)
            batch_probs_list.append(current_batch_probs_np)

    avg_test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else 0

    # Concatenate results from all batches
    y_true = np.concatenate(batch_labels_list) if batch_labels_list else np.array([])
    y_pred = np.concatenate(batch_preds_list) if batch_preds_list else np.array([])
    y_prob = np.concatenate(batch_probs_list) if batch_probs_list else np.array([])
    
    # Ensure y_true and y_pred are 1D for sklearn metrics, if they aren't already
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    # y_prob will be [N_samples, num_classes] for multiclass, or [N_samples, 1] / [N_samples] for binary
    # For binary ROC, it needs to be 1D probabilities of the positive class
    if is_binary and y_prob.ndim == 2 and y_prob.shape[1] == 1:
        y_prob_for_roc = y_prob.squeeze(-1)
    elif is_binary and y_prob.ndim == 1: # Already 1D
        y_prob_for_roc = y_prob
    else: # Multiclass or if y_prob is already in [N, C] format for roc_auc_score
        y_prob_for_roc = y_prob


    if len(y_true) == 0: # Or check y_pred as well
        logger.warning(f"No data to evaluate for {model_name} after processing batches. Lengths: y_true={len(y_true)}, y_pred={len(y_pred)}. Skipping metrics calculation.")
        # Return empty/default metrics
        return {
            'model_name': model_name, 'test_loss': avg_test_loss, 'accuracy': 0, 
            'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0,
            'y_true': [], 'y_pred': [], 'y_prob': [], 'confusion_matrix': [],
            'fpr': [], 'tpr': []
        }

    # --- Metrics Calculation (This line should now be safe) ---
    accuracy = accuracy_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    avg_setting_multiclass = 'macro' # or 'weighted'
    precision, recall, f1, roc_auc = 0, 0, 0, 0
    fpr_list, tpr_list = [], [] # Use lists for fpr, tpr as they might be None/empty

    if not is_binary and num_classes > 1: # Multiclass
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=avg_setting_multiclass, zero_division=0, labels=np.arange(num_classes))
        if len(np.unique(y_true)) > 1 and y_prob_for_roc.shape[0] > 0 and y_prob_for_roc.ndim == 2 and y_prob_for_roc.shape[1] == num_classes:
            try:
                roc_auc = roc_auc_score(y_true, y_prob_for_roc, multi_class='ovr', average=avg_setting_multiclass, labels=np.arange(num_classes))
            except Exception as e_auc:
                logger.warning(f"Could not calculate multiclass AUC for {model_name}: {e_auc}")
        # fpr, tpr remain empty for standard multiclass summary, or you'd calculate per-class
    else: # Binary
        # Ensure labels for binary metrics are [0,1] or whatever is present
        unique_true_labels = np.unique(y_true)
        if len(unique_true_labels) == 0 : # No true labels, should not happen if len(y_true)>0
             precision, recall, f1 = 0,0,0
        elif len(unique_true_labels) == 1: # Only one class in y_true for this batch/dataset
            # f1_score for single class is ill-defined or 0 depending on settings.
            # We can report based on presence of positive class (assuming 1)
            pos_label_binary = 1 
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', pos_label=pos_label_binary, zero_division=0)
            logger.warning(f"Only one class ({unique_true_labels[0]}) present in y_true for binary metrics calculation of {model_name}. Metrics reported for pos_label={pos_label_binary}.")
        else: # Both classes present
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

        if len(unique_true_labels) > 1 and y_prob_for_roc.shape[0] > 0: # Need at least two classes in y_true for roc_curve
            try:
                fpr_vals, tpr_vals, _ = roc_curve(y_true, y_prob_for_roc) # y_prob_for_roc is 1D for binary
                roc_auc = auc(fpr_vals, tpr_vals)
                fpr_list, tpr_list = fpr_vals.tolist(), tpr_vals.tolist()
            except ValueError as ve: # Catches "Only one class present in y_true" from roc_curve
                 logger.warning(f"ValueError calculating binary ROC curve for {model_name} (likely single class in y_true): {ve}")
            except Exception as e_auc:
                logger.warning(f"Could not calculate binary AUC for {model_name}: {e_auc}")
        elif len(unique_true_labels) <= 1 :
             logger.warning(f"Only one class or no classes present in y_true for {model_name}. Cannot calculate ROC AUC.")


    metrics = {
        'model_name': model_name, 'test_loss': avg_test_loss,
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': roc_auc,  'kappa': kappa,
        'y_true': y_true.tolist(), 'y_pred': y_pred.tolist(), 
        'y_prob': y_prob.tolist() if isinstance(y_prob, np.ndarray) else y_prob, # y_prob can be list of lists too
        'confusion_matrix': conf_matrix.tolist(),
        'fpr': fpr_list, # Already a list or empty list
        'tpr': tpr_list  # Already a list or empty list
    }

    logger.info(f"{model_name} Test Set - Loss: {avg_test_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}, Kappa: {kappa:.4f}")

    if report_dir and subdirs:
        figures_subdir = subdirs.get('figures')
        # metrics_subdir = subdirs.get('metrics')
        if not figures_subdir :
            logger.warning(f"Figures subdirectory not found for {model_name}. Plots will not be saved.")
        else:
            # Save ROC curve
            if fpr_list and tpr_list : # Only plot if data is available
                fig_roc = plot_roc_curve([fpr_list], [tpr_list], [roc_auc], [model_name]) # plot_roc_curve expects lists of lists/arrays
                save_figure(fig_roc, f"{model_name}_roc_curve", figures_subdir) #subdirs['figures'])
            else:
                logger.info(f"ROC curve not plotted for {model_name} due to insufficient data for FPR/TPR.")
                
            if metrics.get('confusion_matrix') is not None:
                    cm_array_for_plot = np.array(metrics['confusion_matrix'])
                    
                    # Determine class labels for the confusion matrix plot
                    actual_cm_labels = class_labels_for_plots # Use passed labels
                    if actual_cm_labels is None or len(actual_cm_labels) != cm_array_for_plot.shape[0]:
                        logger.warning(f"Provided class_labels_for_plots is invalid for {model_name}. Generating generic labels.")
                        actual_cm_labels = [f"C{i}" for i in range(cm_array_for_plot.shape[0])]
                    
                    fig_cm = plot_confusion_matrix(cm_array_for_plot, model_name, class_labels=actual_cm_labels)
                    save_figure(fig_cm, f"{model_name}_confusion_matrix", figures_subdir)
                    logger.info(f"Confusion matrix plot saved for {model_name} in {figures_subdir}")
            else: logger.info(f"Confusion matrix not available to plot for {model_name}")

       
    return metrics




# --- 3. VISUALIZATION FUNCTIONS ---
def plot_training_curves(train_losses, val_losses, model_name):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt.gcf()

def plot_confusion_matrix(conf_matrix, model_name,  class_labels=None):
    # Ensure conf_matrix is a numpy array
    cm_array = np.array(conf_matrix)
    
    if class_labels is None:
        # Default to generic labels if not provided, based on matrix shape
        class_labels = [f"Class {i}" for i in range(cm_array.shape[0])]
        if cm_array.shape[0] == 2: # Common binary case
             class_labels = ['Class 0', 'Class 1']
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, # Use dynamic labels
                yticklabels=class_labels)
                # xticklabels=['Class 0', 'Class 1'],
                # yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{model_name} Confusion Matrix')
    plt.tight_layout()
    return plt.gcf()

def plot_roc_curve(fpr_list, tpr_list, auc_list, model_names_list):
    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, (fpr, tpr, auc_val, name) in enumerate(zip(fpr_list, tpr_list, auc_list, model_names_list)):
        if fpr is not None and tpr is not None and len(fpr) > 0 and len(tpr) > 0 :
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                     label=f'{name} (AUC = {auc_val:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt.gcf()


def plot_circular_shap_heatmap(shap_values_array, group_metrics, num_zones_per_metric, model_name_prefix, shap_output_dir, plot_title, filename_suffix, is_global):
    """
    Generates a circular SHAP heatmap for both global and instance plots.
    This version creates a composite image with a subplot for each feature group.
    """
    logger.info(f"Generating circular SHAP heatmap: {plot_title}")
    
    total_shap_features = len(shap_values_array)
    expected_features = len(group_metrics) * num_zones_per_metric
    
    if total_shap_features != expected_features:
        logger.error(f"Cannot create circular heatmap '{plot_title}'. Feature count mismatch. Expected {expected_features}, but got {total_shap_features}.")
        return
        
    num_metrics = len(group_metrics)
    try:
        heatmap_data = pd.DataFrame(shap_values_array.reshape(num_metrics, num_zones_per_metric), index=group_metrics)
    except ValueError as e:
        logger.error(f"Failed to reshape SHAP values for heatmap '{plot_title}': {e}")
        return

    # Define annuli geometry
    r_inner, r_middle, r_outer = 0.3, 0.6, 1.0
    zone_definitions = []
    
    angles_inner = np.linspace(0, 360, 4 + 1)
    for i in range(4): zone_definitions.append({"data_col_idx": i, "display_label": str(i + 1), "r": r_inner, "theta1": angles_inner[i], "theta2": angles_inner[i+1], "annulus_width": r_inner})
    
    angles_middle = np.linspace(0, 360, 8 + 1)
    for i in range(8): zone_definitions.append({"data_col_idx": i + 4, "display_label": str(i + 5), "r": r_middle, "theta1": angles_middle[i], "theta2": angles_middle[i+1], "annulus_width": r_middle-r_inner})

    angles_outer = np.linspace(0, 360, 8 + 1)
    for i in range(8): zone_definitions.append({"data_col_idx": i + 12, "display_label": str(i + 13), "r": r_outer, "theta1": angles_outer[i], "theta2": angles_outer[i+1], "annulus_width": r_outer-r_middle})

    # Setup composite plot
    ncols = 3
    nrows = int(np.ceil(num_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows + 0.7))
    master_axes_flat = axes.flatten() if num_metrics > 1 else [axes]

    # Setup colormap and normalizer
    if is_global:
        cmap, norm = plt.cm.viridis, colors.Normalize(vmin=heatmap_data.values.min(), vmax=heatmap_data.values.max())
        cbar_label = "Mean Abs SHAP Value"
    else:
        vmax = np.abs(heatmap_data.values).max()
        if vmax == 0: vmax = 1 # Avoid error with all-zero SHAP values
        cmap, norm = plt.cm.RdBu_r, colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        cbar_label = "SHAP Value"
        
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for feature_idx, feature_name in enumerate(heatmap_data.index):
        if feature_idx >= len(master_axes_flat): break
        ax = master_axes_flat[feature_idx]
        ax.set_aspect('equal', adjustable='box')
        
        for zone_def in zone_definitions:
            if zone_def["data_col_idx"] < heatmap_data.shape[1]:
                shap_val = heatmap_data.iloc[feature_idx, zone_def["data_col_idx"]]
                color = cmap(norm(shap_val))
                
                # Convert math angles (CCW from East) to plotting angles (CW from North)
                wedge_theta1_deg = (90 - zone_def["theta2"] + 360) % 360
                wedge_theta2_deg = (90 - zone_def["theta1"] + 360) % 360
                
                wedge = patches.Wedge(center=(0, 0), r=zone_def['r'], theta1=wedge_theta1_deg, theta2=wedge_theta2_deg, width=zone_def['annulus_width'], facecolor=color, edgecolor='black', linewidth=0.3)
                ax.add_patch(wedge)
                
                label_angle_rad = np.deg2rad(90 - (zone_def["theta1"] + zone_def["theta2"]) / 2)
                label_radius = zone_def['r'] - zone_def['annulus_width'] / 2
                x_label = label_radius * np.cos(label_angle_rad)
                y_label = label_radius * np.sin(label_angle_rad)
                ax.text(x_label, y_label, zone_def["display_label"], ha='center', va='center', fontsize=8, color="white" if norm(shap_val) < 0.4 else "black")

        ax.set_xlim(-r_outer - 0.1, r_outer + 0.1)
        ax.set_ylim(-r_outer - 0.1, r_outer + 0.1)
        ax.axis('off')
        ax.set_title(feature_name, fontsize=12)

    for i in range(num_metrics, len(master_axes_flat)):
        master_axes_flat[i].axis('off')

    fig.suptitle(plot_title, fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(sm, cax=cbar_ax, label=cbar_label)
    
    if shap_output_dir:
        save_path = Path(shap_output_dir) / f"{model_name_prefix}_shap_circular_{filename_suffix}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info(f"Saved circular SHAP heatmap to {save_path}")
    else:
        plt.show()
 


def run_shap_analysis_dl(
    model,
    X_data_np,
    X_train_data_np_for_background,
    model_name_prefix="model",
    subdirs=None,
    feature_names=None,
    is_binary_mode_flag=True,
    num_classes_for_multiclass=None,
    is_transformer_model_heuristic_for_shap=False,
    shap_num_samples_cli=10, #50, # from parsed_args.shap_num_samples
    group_metrics_list=None,
    num_zones_per_metric=None,
    model_expects_channel_dim_func=None,
    device=None,
    label_encoder_for_class_names=None,
    class_indices_to_explain= None, # For multiclass, which classes to explain
    explainer_type='kernel' # 'deep', or 'kernel'.
):
    """
    Runs SHAP analysis on a trained PyTorch model. This is a final, robust version that
    unifies Gradient, Deep, and Kernel explainers to work with a single plotting pipeline.
    """
    logger = logging.getLogger()

    # --- 1. Initial Checks and Setup ---
    if X_data_np.shape[0] == 0:
        logger.warning(f"SHAP analysis for {model_name_prefix} skipped: X_data_np is empty.")
        return None, None

    if X_train_data_np_for_background is None or X_train_data_np_for_background.shape[0] == 0:
        logger.error(f"SHAP analysis for {model_name_prefix} failed: X_train_data_np_for_background is missing or empty.")
        return None, None
    
    if model_expects_channel_dim_func is None or device is None:
        logger.error("model_expects_channel_dim_func or device not provided to run_shap_analysis_dl for {model_name_prefix}.")
        return None, None

    logger.info(f"--- Starting SHAP Analysis for {model_name_prefix} using '{explainer_type}' explainer ---")
    logger.info(f"Explaining {X_data_np.shape[0]} samples. Using background data of shape {X_train_data_np_for_background.shape}")
    
    
    explainer = None
    raw_shap_values = None
    shap_values_arrays = None
    base_values_arrays = None

    # --- 2. Explainer Initialization and Value Calculation ---
    X_data_tensor = torch.tensor(X_data_np, dtype=torch.float32).to(device)
    if model_expects_channel_dim_func(model) and len(X_data_tensor.shape) == 2:
        X_data_tensor = X_data_tensor.unsqueeze(1)

    if explainer_type in ['deep', 'gradient']:
        if X_train_data_np_for_background.shape[0] < 200:
            background_tensor = torch.tensor(X_train_data_np_for_background, dtype=torch.float32).to(device)
        else:
            rand_indices = np.random.choice(X_train_data_np_for_background.shape[0], 200, replace=False)
            background_tensor = torch.tensor(X_train_data_np_for_background[rand_indices], dtype=torch.float32).to(device)
        
        if model_expects_channel_dim_func(model) and len(background_tensor.shape) == 2:
            background_tensor = background_tensor.unsqueeze(1)

        if explainer_type == 'deep':
            logger.info("Using DeepExplainer.")
            explainer = shap.DeepExplainer(model, background_tensor)
            shap_values_arrays = explainer.shap_values(X_data_tensor)
            base_values_arrays = explainer.expected_value
        else: # gradient
            logger.info("Using GradientExplainer.")
            explainer = shap.GradientExplainer(model, background_tensor)
            shap_values_arrays = explainer.shap_values(X_data_tensor)
            with torch.no_grad():
                base_values_arrays = torch.mean(model(background_tensor), dim=0).cpu().numpy()
    
    ## NEWLY RESTORED KERNEL EXPLAINER LOGIC ##
    elif explainer_type == 'kernel':
        logger.warning(f"Using KernelExplainer for {type(model).__name__}. This can be slow and memory-intensive.")

        # ------
        # This wrapper function handles the batch size issue for models with BatchNorm layers.
        def _predict_logits_for_kernel(data_np):
            model.eval()
            original_batch_size = data_np.shape[0]

            # If batch size is 1, duplicate the sample to avoid BatchNorm errors
            if original_batch_size == 1:
                data_np = np.repeat(data_np, 2, axis=0)

            with torch.no_grad():
                data_tensor = torch.tensor(data_np, dtype=torch.float32).to(device)

                # This logic correctly adds a dimension for CNNs/Transformers
                if model_expects_channel_dim_func(model) and len(data_tensor.shape) == 2:
                    data_tensor = data_tensor.unsqueeze(1)

                predictions = model(data_tensor).cpu().numpy()

            # If we duplicated the batch, return only the prediction for the first (original) sample
            return predictions[0:1] if original_batch_size == 1 else predictions
        # ------

        # Summarize background data to keep KernelExplainer manageable
        # num_summary_samples = min(50, X_train_data_np_for_background.shape[0])
        num_summary_samples = min(shap_num_samples_cli, X_train_data_np_for_background.shape[0])
        logger.info(f"Summarizing background data to {num_summary_samples} samples for KernelExplainer.")
        background_summary = shap.kmeans(X_train_data_np_for_background, num_summary_samples).data

        torch.cuda.empty_cache() # Free memory before running the intensive explainer

        # Initialize the explainer with the new, robust prediction function
        explainer = shap.KernelExplainer(_predict_logits_for_kernel, background_summary)
        raw_shap_values = explainer.shap_values(X_data_np)
        shap_values_arrays = explainer.shap_values(X_data_np)
        base_values_arrays = explainer.expected_value

    else:
        logger.error(f"Unsupported explainer_type: '{explainer_type}'.")
        return None, None

    # final Explanation object creation block ---
    if shap_values_arrays is None or base_values_arrays is None:
        logger.error("SHAP values could not be calculated. Aborting.")
        return None, None
    
    # create a SHAP Explanation object
    raw_shap_values = shap.Explanation(
        values=shap_values_arrays,
        base_values=base_values_arrays,
        data=X_data_np,
        feature_names=feature_names
    )


    # --- 3. Prepare Output Directory ---
    shap_output_dir = subdirs['figures'] / "SHAP" / model_name_prefix
    shap_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"SHAP plots will be saved in: {shap_output_dir}")

    # # --- 4. Normalize SHAP outputs into a list of Explanation objects 

    def _squeeze_and_scalarize(values, base_values):
        """Helper to fix shapes from different explainers."""
        # Squeeze 3D values (e.g., from a CNN) into 2D
        if isinstance(values, np.ndarray) and values.ndim == 3 and values.shape[1] == 1:
            values = np.squeeze(values, axis=1)

        # Ensure base_values is a simple number, not an array like [0.04]
        if hasattr(base_values, '__len__') and len(base_values) == 1:
            base_values = base_values[0]

        return values, base_values

    explanations_to_plot = []
    class_names = []

    if isinstance(raw_shap_values.values, list): # Multiclass or Binary-with-2-outputs
        logger.info("Model returned multiple outputs. Creating explanations for each class.")
        for i, class_values in enumerate(raw_shap_values.values):
            sanitized_values, sanitized_base_values = _squeeze_and_scalarize(class_values, raw_shap_values.base_values[i])
            class_name = label_encoder_for_class_names.inverse_transform([i])[0] if label_encoder_for_class_names else f"Class {i}"

            explanations_to_plot.append(shap.Explanation(
                values=sanitized_values, base_values=sanitized_base_values,
                data=X_data_np, feature_names=feature_names
            ))
            class_names.append(class_name)
    else: # Binary-with-1-output
        logger.info("Model returned a single output. Creating explanations for Class 1 and Class 0.")

        # Sanitize the single output first
        sanitized_values, sanitized_base_values = _squeeze_and_scalarize(raw_shap_values.values, raw_shap_values.base_values)

        # Create a clean Explanation object for Class 1
        expl_c1 = shap.Explanation(
            values=sanitized_values, base_values=sanitized_base_values,
            data=X_data_np, feature_names=feature_names
        )
        explanations_to_plot.append(expl_c1)
        class_names.append("Class 1")

        # Create the explanation for Class 0 by negating the sanitized, scalar base value
        expl_c0 = shap.Explanation(
            values=-sanitized_values, base_values=-sanitized_base_values,
            data=X_data_np, feature_names=feature_names
        )
        explanations_to_plot.append(expl_c0)
        class_names.append("Class 0")
    
    # explanations_to_plot = []
    # class_names = []
    
    # # We now use the explainer object itself to get consistent Explanation objects
    # if not isinstance(raw_shap_values, shap.Explanation):
    #      # KernelExplainer may return raw arrays; convert to Explanation object
    #      raw_shap_values = shap.Explanation(
    #          values=raw_shap_values,
    #          base_values=explainer.expected_value,
    #          data=X_data_np,
    #          feature_names=feature_names
    #      )

    # if isinstance(raw_shap_values.values, list): # Multiclass or Binary-with-2-outputs
    #     logger.info("Model returned multiple outputs. Creating explanations for each class.")
    #     for i, class_values in enumerate(raw_shap_values.values):
    #         class_name = label_encoder_for_class_names.inverse_transform([i])[0] if label_encoder_for_class_names else f"Class {i}"
    #         base_val_for_class = raw_shap_values.base_values[i] if hasattr(raw_shap_values.base_values, '__len__') else raw_shap_values.base_values
    #         explanations_to_plot.append(shap.Explanation(
    #             values=class_values, base_values=base_val_for_class,
    #             data=X_data_np, feature_names=feature_names
    #         ))
    #         class_names.append(class_name)
    # else: # Binary-with-1-output (less common for deep learning but supported)
    #     logger.info("Model returned a single output. Creating explanations for Class 1 (Positive) and Class 0 (Negative).")
    #     explanations_to_plot.append(raw_shap_values)
    #     class_names.append("Class 1")
    #     expl_c0 = shap.Explanation(
    #         values=-raw_shap_values.values, base_values=-raw_shap_values.base_values,
    #         data=X_data_np, feature_names=feature_names
    #     )
    #     explanations_to_plot.append(expl_c0)
    #     class_names.append("Class 0")

    # --- 5. Pre-calculate All Logits and Probabilities for Titles ---
    # This is more efficient and robust for all plot types.
    all_logits = np.array([
        (exp.base_values + exp.values.sum(axis=1)) for exp in explanations_to_plot
    ]).T # Transpose to get shape (n_samples, n_classes)
    
    all_softmax_probs = softmax(all_logits, axis=1)
    
 
    # --- 6. MASTER PLOTTING LOOP ---
    for i, expl in enumerate(explanations_to_plot):
        class_name = class_names[i]
        class_name_for_file = str(class_name).replace(' ', '_')
        
        logger.info(f"--- Generating all plots for: {class_name} ---")

        # --- Summary and Bar Plots ---
        try:
            plt.figure()
            shap.summary_plot(expl.values, X_data_np, feature_names=feature_names, show=False)
            plt.title(f"SHAP Summary - {class_name}")
            plt.savefig(shap_output_dir / f"{model_name_prefix}_summary_{class_name_for_file}.png", bbox_inches='tight')
            plt.close()

            plt.figure()
            shap.plots.bar(expl, show=False)
            plt.title(f"SHAP Feature Importance - {class_name}")
            plt.savefig(shap_output_dir / f"{model_name_prefix}_bar_{class_name_for_file}.png", bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Failed to generate summary/bar plots for {class_name}: {e}")

        # --- Waterfall and Grouped Waterfall Plots ---
        instances_to_plot = [0, 1, 2, 3, 4, 9]
        for inst_idx in instances_to_plot:
            if inst_idx >= len(X_data_np): continue
            
            try:
                # Get the pre-calculated probability for the title
                probability = all_softmax_probs[inst_idx, i]
                
                title = (f"SHAP Waterfall (Logit Space) - Inst {inst_idx}, {class_name}\n"
                         f"{model_name_prefix} | Certainty: {probability:.1%}")
                filename = f"{model_name_prefix}_waterfall_LOGIT_inst_{inst_idx}_{class_name_for_file}.png"
                
                plt.figure()
                shap.plots.waterfall(expl[inst_idx], max_display=15, show=False)
                plt.title(title)
                plt.savefig(shap_output_dir / filename, bbox_inches='tight')
                plt.close()

                if group_metrics_list and num_zones_per_metric:
                    instance_explanation = expl[inst_idx]
                    grouped_vals = [instance_explanation.values[j*num_zones_per_metric:(j+1)*num_zones_per_metric].sum() for j in range(len(group_metrics_list))]
                    grouped_expl = shap.Explanation(
                        values=np.array(grouped_vals), base_values=instance_explanation.base_values, feature_names=group_metrics_list
                    )
                    grouped_title = f"Grouped SHAP Waterfall - Inst {inst_idx}, {class_name}"
                    grouped_filename = f"{model_name_prefix}_grouped_waterfall_inst_{inst_idx}_{class_name_for_file}.png"
                    plt.figure()
                    shap.plots.waterfall(grouped_expl, show=False)
                    plt.title(grouped_title)
                    plt.savefig(shap_output_dir / grouped_filename, bbox_inches='tight')
                    plt.close()
                else:
                    logger.info(f"Skipping grouped waterfall plot for {class_name} due to missing group metrics or zones.")
                

            except Exception as e:
                logger.error(f"Failed to generate waterfall plot for instance {inst_idx}, class {class_name}: {e}")

            # --- Grouped Heatmap, Bar, and Circular Plots ---
        if group_metrics_list and num_zones_per_metric and \
           feature_names and len(feature_names) == (len(group_metrics_list) * num_zones_per_metric):
            try:
                logger.info(f"Generating grouped plots for {class_name}...")
                # The SHAP values for the current class are taken directly from the explanation object.
                shap_values_for_class = expl.values
                num_metrics = len(group_metrics_list)

                # --- 1. Grouped SHAP Heatmap (Clustermap) ---
                mean_abs_shap_reshaped = np.mean(np.abs(shap_values_for_class), axis=0).reshape(num_metrics, num_zones_per_metric)
                heatmap_data = pd.DataFrame(mean_abs_shap_reshaped, index=group_metrics_list,
                                            columns=[f"Z{z+1}" for z in range(num_zones_per_metric)])

                if heatmap_data.shape[0] > 1 and heatmap_data.shape[1] > 1:
                    hm_cluster = sns.clustermap(heatmap_data, annot=True, fmt=".3f", cmap="viridis",
                                                figsize=(max(8, num_zones_per_metric * 0.6), max(6, num_metrics * 0.5)),
                                                linewidths=.5)
                    plt.setp(hm_cluster.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
                    # Use the specific class name for the title
                    hm_cluster.fig.suptitle(f"Mean Abs SHAP Clustermap ({class_name}) - {model_name_prefix}", y=1.03, fontsize=10)
                    plt.savefig(shap_output_dir / f"{model_name_prefix}_shap_clustermap_{class_name_for_file}.png", bbox_inches='tight')
                    plt.close(hm_cluster.fig)
                else:
                    logger.warning(f"Not enough data to create a clustermap for {class_name}. Heatmap shape: {heatmap_data.shape}")

                # --- 2. Aggregated Feature Group Importance Bar Plot ---
                metric_aggregated_importance = np.sum(mean_abs_shap_reshaped, axis=1)
                sorted_indices = np.argsort(metric_aggregated_importance)[::-1]

                fig_grouped_bar, ax_grouped_bar = plt.subplots(figsize=(max(8, num_metrics * 0.8), 6))
                ax_grouped_bar.bar(np.array(group_metrics_list)[sorted_indices], metric_aggregated_importance[sorted_indices])
                plt.xticks(rotation=45, ha="right")
                ax_grouped_bar.set_ylabel("Sum of Mean(|SHAP value|) across Zones")
                ax_grouped_bar.set_title(f"Feature Group Importance ({class_name}) - {model_name_prefix}", fontsize=12)
                plt.tight_layout()
                plt.savefig(shap_output_dir / f"{model_name_prefix}_feature_group_importance_{class_name_for_file}.png", dpi=150)
                plt.close(fig_grouped_bar)

                # --- 3. Circular Heatmaps (Composite Image) ---
                if not heatmap_data.empty:
                    logger.info(f"Generating composite circular SHAP heatmap for {class_name}...")

                    # Define radii for the rings
                    r_inner = 0.3 
                    r_middle = 0.6
                    r_outer = 1.0


                    new_zone_definitions = []
                    
                    start_segment_offsets = {
                            "inner": 0,  # Label 1 starts at geometric segment 0 (NE)
                            "middle": 1, # Label 5 starts at geometric segment 1 (the one just CW from NE)
                            "outer": 1   # Label 13 starts at geometric segment 1
                        }

                    # Innermost Ring (Target Display Labels: 1 (NE), 2 (NW), 3 (SW), 4 (SE))
                    num_segments_inner = 4
                    angles_inner_vad = np.linspace(0, 360, num_segments_inner + 1) # VAD: [0, 90, 180, 270, 360]
                    base_label_inner = 1
                    target_start_idx_inner = start_segment_offsets["inner"]
                    for i in range(num_segments_inner): # i is the geometric segment index (0 to N-1), CW from North
                        k_raw = target_start_idx_inner - i
                        # k is the label sequence index (0 for base_label, 1 for base_label+1, etc. in CCW order)
                        k = (k_raw % num_segments_inner + num_segments_inner) % num_segments_inner
                        actual_display_label_val = base_label_inner + k
                        
                        new_zone_definitions.append({
                            "data_col_name": f"Z{actual_display_label_val}",
                            "display_label": str(actual_display_label_val),
                            "r": r_inner,
                            "theta1": angles_inner_vad[i],
                            "theta2": angles_inner_vad[i+1],
                            "annulus_width": r_inner
                        })

                    # Middle Ring (Target Display Labels: 5 (NE-most), 6 (next CCW), ..., 12)
                    num_segments_middle = 8
                    angles_middle_vad = np.linspace(0, 360, num_segments_middle + 1)
                    base_label_middle = 5
                    target_start_idx_middle = start_segment_offsets["middle"]
                    for i in range(num_segments_middle):
                        k_raw = target_start_idx_middle - i
                        k = (k_raw % num_segments_middle + num_segments_middle) % num_segments_middle
                        actual_display_label_val = base_label_middle + k
                        
                        new_zone_definitions.append({
                            "data_col_name": f"Z{actual_display_label_val}",
                            "display_label": str(actual_display_label_val),
                            "r": r_middle,
                            "theta1": angles_middle_vad[i],
                            "theta2": angles_middle_vad[i+1],
                            "annulus_width": r_middle - r_inner
                        })

                    # Outer Ring
                    num_segments_outer = 8
                    angles_outer_vad = np.linspace(0, 360, num_segments_outer + 1)
                    base_label_outer = 13
                    target_start_idx_outer = start_segment_offsets["outer"]
                    for i in range(num_segments_outer):
                        k_raw = target_start_idx_outer - i
                        k = (k_raw % num_segments_outer + num_segments_outer) % num_segments_outer
                        actual_display_label_val = base_label_outer + k
                            
                        new_zone_definitions.append({
                            "data_col_name": f"Z{actual_display_label_val}",
                            "display_label": str(actual_display_label_val),
                            "r": r_outer,
                            "theta1": angles_outer_vad[i],
                            "theta2": angles_outer_vad[i+1],
                            "annulus_width": r_outer - r_middle
                        })
                    zone_definitions = new_zone_definitions

        ###########  generate composite image with all 9 feat HM circular plots ---

                    num_features = len(heatmap_data.index)
                    if num_features == 0:
                        logger.info("No features to plot in composite image.")
                    #else: # Proceed to plot if there are features

                    # Determine grid size (defaulting to 3x3 for up to 9 features)
                    ncols = 3
                    nrows = int(np.ceil(num_features / ncols))
                    if num_features == 0: # handle case of no features
                        logger.info("Skipping composite plot as there are no features.")
                    else:
                        # Adjust figsize: Aim for each subplot to be reasonably clear.
                        # If each subplot needs roughly 4-5 inches, a 3x3 grid could be 12-15 inches wide.
                        # Height needs to accommodate suptitle.
                        master_fig_width = 4.5 * ncols 
                        master_fig_height = 4.5 * nrows + 0.7 # Add a bit more for suptitle

                        master_fig, master_axes = plt.subplots(nrows, ncols, figsize=(master_fig_width, master_fig_height))
                        
                        # If nrows=1 and ncols=1, master_axes is not an array but a single Axes.
                        # If nrows=1 or ncols=1 (but not both 1), master_axes is a 1D array.
                        # Otherwise, it's a 2D array. Flatten for consistent iteration.
                        if num_features == 1:
                            master_axes_flat = [master_axes] # Make it iterable
                        else:
                            master_axes_flat = master_axes.flatten()

                        # Global normalization for a shared colorbar (already done for individual plots, reuse here)
                        # Assuming g_vmin, g_vmax, norm, cmap, sm are already defined globally from single plot logic
                        # If not, recalculate them here:
                        g_vmin = heatmap_data.min().min()
                        g_vmax = heatmap_data.max().max()
                        if g_vmin == g_vmax:
                            if g_vmin == 0: g_vmin, g_vmax = -0.001, 0.001
                            else: 
                                abs_g_vmin = abs(g_vmin) # Calculate absolute value once
                                g_vmin = g_vmin - abs_g_vmin * 0.1 if g_vmin != 0 else -0.001
                                g_vmax = g_vmax + abs_g_vmin * 0.1 if g_vmax != 0 else 0.001 # Use abs_g_vmin here too for consistency
                            if g_vmin == g_vmax: g_vmin -=0.001; g_vmax +=0.001 
                                    
                        current_norm = plt.Normalize(vmin=g_vmin, vmax=g_vmax) # Use current_norm for clarity
                        current_cmap = plt.cm.viridis # Or your chosen cmap
                        sm = plt.cm.ScalarMappable(cmap=current_cmap, norm=current_norm)
                        sm.set_array([])

                        # Loop through each feature and plot on a subplot
                        for feature_idx, feature_name in enumerate(heatmap_data.index):
                            if feature_idx >= len(master_axes_flat):
                                logger.warning(f"Feature index {feature_idx} exceeds available subplots. Skipping.")
                                break
                                
                            ax = master_axes_flat[feature_idx] # Get the current subplot axes

                            # --- Core Cartesian Drawing Logic for one circular plot ---
                            ax.set_aspect('equal', adjustable='box')
                            current_feature_shap_values = heatmap_data.loc[feature_name]

                            for zone_def in zone_definitions: # Use the reordered zone_definitions
                                shap_value = current_feature_shap_values[zone_def["data_col_name"]]
                                color = current_cmap(current_norm(shap_value)) # Use global norm and cmap

                                r_outer_segment = zone_def["r"]
                                r_inner_segment = zone_def["r"] - zone_def["annulus_width"]
                                vad_start = zone_def["theta1"]
                                vad_end = zone_def["theta2"]
                                wedge_theta1_deg = (90 - vad_end + 360) % 360
                                wedge_theta2_deg = (90 - vad_start + 360) % 360

                                wedge_shape = patches.Wedge(
                                    center=(0, 0), r=r_outer_segment,
                                    theta1=wedge_theta1_deg, theta2=wedge_theta2_deg,
                                    width=r_outer_segment - r_inner_segment
                                )
                                path_patch = patches.PathPatch(
                                    wedge_shape.get_path(), facecolor=color,
                                    edgecolor='black', linewidth=0.3 # Thinner lines for subplots
                                )
                                ax.add_patch(path_patch)

                                label_vad_center = (vad_start + vad_end) / 2
                                label_radius = r_outer_segment - (r_outer_segment - r_inner_segment) / 2
                                if r_inner_segment == 0: label_radius = r_outer_segment / 1.6
                                label_mad_rad = np.deg2rad((90 - label_vad_center + 360) % 360)
                                x_label = label_radius * np.cos(label_mad_rad)
                                y_label = label_radius * np.sin(label_mad_rad)
                                ax.text(x_label, y_label, zone_def["display_label"],
                                        ha='center', va='center', fontsize=5, # Smaller font for subplots
                                        color="white" if current_norm(shap_value) < 0.3 else "black")

                            plot_radius_padding = 0.05
                            ax.set_xlim(-r_outer - plot_radius_padding, r_outer + plot_radius_padding)
                            ax.set_ylim(-r_outer - plot_radius_padding, r_outer + plot_radius_padding)
                            ax.axis('off')
                            ax.set_title(feature_name, fontsize=9) # Title for each subplot
                            # --- End of Core Cartesian Drawing Logic ---

                        # Turn off any unused subplots if num_features isn't a perfect multiple of nrows*ncols
                        for i in range(num_features, nrows * ncols):
                            if i < len(master_axes_flat): # Check index bounds
                                master_axes_flat[i].axis('off')

                        # Add a single, shared colorbar to the master figure
                        # Adjust [left, bottom, width, height] as needed for your master_fig_width/height
                        # These are fractions of the *figure* dimensions.
                        cbar_left = 0.91 # Positioned towards the right
                        cbar_bottom = 0.15 
                        cbar_width = 0.015 # Thin colorbar
                        cbar_height = 0.7 # Cover 70% of figure height vertically
                        
                        # Check if figure has space for colorbar based on calculated width/height
                        if master_fig_width * (1 - cbar_left - cbar_width) < 0.5: # Heuristic: if not enough space
                            logger.warning("Figure might be too narrow for the colorbar at specified position. Adjusting.")
                            cbar_left = 0.88 # Try to pull it left a bit if figure is very narrow

                        cbar_ax = master_fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
                        master_fig.colorbar(sm, cax=cbar_ax, orientation='vertical', label="Mean Abs SHAP")

                        # Add overall title to the master figure
                        title_detail_composite = 'Positive Class' if is_binary_mode_flag else 'Class 0'
                        master_fig.suptitle(f"All Features - Circular SHAP Heatmaps\n(Mean Abs SHAP for {title_detail_composite} - {model_name_prefix})",
                                            fontsize=14, y=0.98) # Adjust y if suptitle overlaps subplots

                        # Adjust layout to prevent overlap, considering manually placed colorbar
                        # rect can be [left, bottom, right, top]
                        master_fig.tight_layout(rect=[0, 0.03, cbar_left - 0.02, 0.95]) # Leave space for cbar and suptitle

                        # Save the master figure
                        master_fig_filename = shap_output_dir / f"{model_name_prefix}_shap_feat_imp_circ.png"
                        master_fig.savefig(master_fig_filename, dpi=150) # Use a reasonable DPI
                        plt.close(master_fig) # Close the master figure
                        logger.info(f"Saved composite circular SHAP heatmap to: {master_fig_filename}")
                else:
                    logger.warning(f"Skipping circular SHAP heatmap for {class_name}: Heatmap data is empty.")
        

            except Exception as e:
                    logger.error(f"Failed during grouped plot generation for {class_name}: {e}")
        else:
            logger.info(f"Skipping grouped plots for {class_name}: Conditions not met (check feature counts and group metrics).")
        
    # elif group_metrics_list and num_zones_per_metric :
    #      logger.warning(f"Skipping grouped SHAP plot for {model_name_prefix}: Feature count mismatch.")
         
    # else:
    #     logger.info(f"Skipping all grouped SHAP plots for {model_name_prefix} due to initial condition failure.")
    
    # return explainer_shap_values_list, explainer
     
    logger.info(f"SHAP analysis for {model_name_prefix} (potentially) completed.")
    return explanations_to_plot, explainer

# FOr OFA dataset, paper work using kernelExplainer
def run_shap_analysis_dl_OFA(model,
                         X_data_np, # Data to explain
                         X_train_data_np_for_background, # Background data for SHAP
                         model_name_prefix="model",
                         subdirs=None,
                         feature_names=None,
                         is_binary_mode_flag=True,
                         num_classes_for_multiclass=None,
                         is_transformer_model_heuristic_for_shap=False,
                         shap_num_samples_cli=10, #50, # Value from parsed_args.shap_num_samples
                         group_metrics_list=None,
                         num_zones_per_metric=None,
                         model_expects_channel_dim_func=None, # The actual helper function
                         device=None,
                         label_encoder_for_class_names=None,
                         class_indices_to_explain=None, 
                         explainer_type=None, 
                         ):
    """
    Runs SHAP analysis on a trained PyTorch model.
    Uses X_data_np for the data to explain, consistent with calls.
    """
    if X_data_np.shape[0] == 0:
        logger.warning(f"SHAP analysis for {model_name_prefix} skipped: X_data_np is empty.")
        return None, None

    if X_train_data_np_for_background is None or X_train_data_np_for_background.shape[0] == 0:
        logger.error(f"SHAP analysis for {model_name_prefix} failed: X_train_data_np_for_background is missing or empty.")
        return None, None
    
    if model_expects_channel_dim_func is None or device is None:
        logger.error("model_expects_channel_dim_func or device not provided to run_shap_analysis_dl for {model_name_prefix}.")
        return None, None

    logger.info(f"--- Starting SHAP Analysis for {model_name_prefix} ---")
    logger.info(f"Explaining {X_data_np.shape[0]} samples. Using background data of shape {X_train_data_np_for_background.shape}")

    # --- SHAP STABILITY: Background Data Handling ---
    # For very small datasets (like 83 training samples from previous logs), using all training data is often best.
    # The variable `always_use_full_background` can be set based on dataset size or as a fixed strategy.
    # Let's assume if the background dataset passed is small, we use it all.
    use_full_background_heuristic = X_train_data_np_for_background.shape[0] < 150 # Example heuristic

    if is_transformer_model_heuristic_for_shap or "cnntransformer" in model_name_prefix.lower() or use_full_background_heuristic:
        logger.info(f"SHAP Analysis for {model_name_prefix}: Using all {X_train_data_np_for_background.shape[0]} training samples directly for SHAP background (no k-means).")
        X_background_summary_np = X_train_data_np_for_background
    else:
        # Original k-means logic for larger datasets if preferred
        background_samples_max = shap_num_samples_cli
        if is_transformer_model_heuristic_for_shap: # This condition is now part of the above block
            pass # background_samples_max = min(shap_num_samples_cli, 70)
        else:
            background_samples_max = min(shap_num_samples_cli, 100)
        
        current_background_samples_k = min(background_samples_max, X_train_data_np_for_background.shape[0])
        if current_background_samples_k < 5 and X_train_data_np_for_background.shape[0] >= 5 :
             current_background_samples_k = min(5, X_train_data_np_for_background.shape[0])
        logger.info(f"Summarizing background data for SHAP using k-means with k={current_background_samples_k} from {X_train_data_np_for_background.shape[0]} samples for {model_name_prefix}.")
        if current_background_samples_k > 0 :
            X_background_summary_np = shap.kmeans(X_train_data_np_for_background, current_background_samples_k).data
        else: # Fallback if k is somehow 0
            logger.warning(f"SHAP background k is 0 for {model_name_prefix}. Using single mean sample for background.")
            X_background_summary_np = np.mean(X_train_data_np_for_background, axis=0, keepdims=True)


    if X_background_summary_np is None or X_background_summary_np.shape[0] == 0:
        logger.error(f"SHAP background summary (X_background_summary_np) is empty for {model_name_prefix}. Cannot proceed.")
        return None, None
    
    logger.info(f"Actual SHAP background summary data shape: {X_background_summary_np.shape}")

    def _predict_proba_for_shap_dl(data_np_internal):
        model.eval()
        with torch.no_grad():
            data_tensor = torch.tensor(data_np_internal, dtype=torch.float32).to(device)
            # Critical: model_expects_channel_dim_func checks the *specific model instance*
            if model_expects_channel_dim_func(model):
                 if len(data_tensor.shape) == 2:
                    data_tensor = data_tensor.unsqueeze(1)

            outputs = model(data_tensor)
            if is_binary_mode_flag:
                probs = torch.sigmoid(outputs)
                return probs.cpu().numpy()
            else:
                probs = torch.softmax(outputs, dim=1)
                return probs.cpu().numpy()

    try:
        explainer = shap.KernelExplainer(_predict_proba_for_shap_dl, X_background_summary_np)
    except Exception as e_explainer:
        logger.error(f"Failed to initialize SHAP KernelExplainer for {model_name_prefix}: {e_explainer}")
        logger.error(traceback.format_exc())
        return None, None

    current_nsamples_for_explainer = shap_num_samples_cli
    if is_transformer_model_heuristic_for_shap:
        current_nsamples_for_explainer = min(current_nsamples_for_explainer, 10) #50)
        logger.info(f"Transformer model heuristic: SHAP nsamples for explainer.shap_values capped at {current_nsamples_for_explainer}.")
    
    logger.info(f"Calculating SHAP values for {X_data_np.shape[0]} samples using nsamples={current_nsamples_for_explainer} for {model_name_prefix}...")
    
    # --- SHAP STABILITY: L1 Regularization ---
    l1_reg_val = 'aic' 
    # More aggressive L1 for models prone to instability (transformers, hybrids) or when background is very small vs features
    if is_transformer_model_heuristic_for_shap or "cnntransformer" in model_name_prefix.lower() or \
       (X_background_summary_np.shape[0] < X_data_np.shape[1] * 1.5 and X_data_np.shape[1] > 100) : # Heuristic for small background vs many features
        l1_reg_val = 'num_features(5)' 
        logger.info(f"Using aggressive L1 regularization for SHAP: l1_reg='{l1_reg_val}' for {model_name_prefix}")
    elif X_background_summary_np.shape[0] < X_data_np.shape[1] * 2: 
        l1_reg_val = 'num_features(10)' 
        logger.info(f"Background data is relatively small vs features; using L1 regularization: l1_reg='{l1_reg_val}' for {model_name_prefix}")
    else: 
        l1_reg_val = 'num_features(20)' 
        logger.info(f"Using L1 regularization for SHAP: l1_reg='{l1_reg_val}' for {model_name_prefix}")

    try:
        torch.cuda.empty_cache()
        explainer_shap_values_list = explainer.shap_values(
            X_data_np, 
            nsamples=current_nsamples_for_explainer, 
            l1_reg=l1_reg_val 
        )
    except Exception as e_shap_values:
        logger.error(f"Failed during explainer.shap_values() for {model_name_prefix} with l1_reg='{l1_reg_val}': {e_shap_values}")
        logger.error(traceback.format_exc())
        try: # Fallback with even more aggressive settings
            l1_reg_fallback = 'num_features(1)'
            nsamples_fallback = min(current_nsamples_for_explainer, 2 * X_data_np.shape[1] + 10)
            logger.warning(f"Retrying SHAP values for {model_name_prefix} with l1_reg='{l1_reg_fallback}' and nsamples={nsamples_fallback}...")
            explainer_shap_values_list = explainer.shap_values(
                X_data_np, 
                nsamples=nsamples_fallback,
                l1_reg=l1_reg_fallback
            )
        except Exception as e_shap_values_retry:
            logger.error(f"Retry for explainer.shap_values() also failed for {model_name_prefix}: {e_shap_values_retry}")
            logger.error(traceback.format_exc())
            return None, None
            
    shap_output_dir = subdirs['figures'] / "SHAP" / model_name_prefix
    if "Error_Analysis" in str(model_name_prefix): # Handle error analysis sub-pathing correctly
        # Assuming model_name_prefix for error analysis is like "ModelName_final_best_holdout_misclassified"
        # And subdirs['figures'] for post_analysis is "reports/.../post_analysis/figures"
        # We want SHAP plots in "reports/.../post_analysis/figures/SHAP_Error_Analysis/ModelName_final_best_holdout_misclassified"
        base_shap_error_dir = subdirs['figures'].parent / "SHAP_Error_Analysis" # Go up one from 'figures' then to 'SHAP_Error_Analysis'
        shap_output_dir = base_shap_error_dir / model_name_prefix
        
    shap_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"SHAP plots will be saved in: {shap_output_dir}")

    # --- Plotting logic (binary vs multiclass, summary, waterfall) ---
    if is_binary_mode_flag:
        shap_values_for_plotting = None; expected_value_for_plotting = None
        if isinstance(explainer_shap_values_list, list):
            if not explainer_shap_values_list: logger.error(f"SHAP list empty for binary {model_name_prefix}"); return None,None
            if len(explainer_shap_values_list) == 1:
                shap_values_for_plotting = explainer_shap_values_list[0]
                if isinstance(explainer.expected_value, (list,np.ndarray)) and len(explainer.expected_value)==1: expected_value_for_plotting = explainer.expected_value[0]
                elif not isinstance(explainer.expected_value, (list,np.ndarray)): expected_value_for_plotting = explainer.expected_value
                else: logger.warning(f"Unusual expected_value for binary {model_name_prefix}"); expected_value_for_plotting = explainer.expected_value
            elif len(explainer_shap_values_list) == 2:
                logger.warning(f"SHAP got 2 arrays for binary {model_name_prefix}, using index 1"); shap_values_for_plotting = explainer_shap_values_list[1]
                if isinstance(explainer.expected_value, (list,np.ndarray)) and len(explainer.expected_value)==2: expected_value_for_plotting = explainer.expected_value[1]
                else: logger.warning(f"Unusual expected_value for binary {model_name_prefix}"); expected_value_for_plotting = explainer.expected_value[1] if isinstance(explainer.expected_value, (list,np.ndarray)) and len(explainer.expected_value)==2 else explainer.expected_value
            else: logger.error(f"Unexpected SHAP list length for binary {model_name_prefix}"); return None,None
        elif isinstance(explainer_shap_values_list, np.ndarray): shap_values_for_plotting = explainer_shap_values_list; expected_value_for_plotting = explainer.expected_value
        else: logger.error(f"Unexpected SHAP type for binary {model_name_prefix}"); return None,None

        if shap_values_for_plotting is None: logger.error(f"No SHAP values for binary plot {model_name_prefix}"); return None,None
        if expected_value_for_plotting is None: logger.warning(f"No expected_value for binary {model_name_prefix}")
        if isinstance(expected_value_for_plotting, (list,np.ndarray)) and len(expected_value_for_plotting)>0 : expected_value_for_plotting = expected_value_for_plotting[0]
        elif isinstance(expected_value_for_plotting, (list,np.ndarray)) and len(expected_value_for_plotting)==0 : expected_value_for_plotting = 0.0


        if np.any(np.abs(shap_values_for_plotting) > 1e5): logger.warning(f"Extreme SHAP values for {model_name_prefix} (max abs: {np.max(np.abs(shap_values_for_plotting)):.2e})")
        class_name_for_plot = "Positive Class"
        
        plt.figure()
        shap.summary_plot(shap_values_for_plotting, X_data_np, feature_names=feature_names, show=False, plot_type="dot")
        plt.title(f"SHAP Summary ({class_name_for_plot}) - {model_name_prefix}")
        plt.tight_layout()
        plt.savefig(shap_output_dir / f"{model_name_prefix}_shap_summary.png", bbox_inches='tight'); plt.close()
        
        plt.figure()
        shap.summary_plot(shap_values_for_plotting, X_data_np, feature_names=feature_names, show=False, plot_type="bar"); plt.title(f"SHAP Bar ({class_name_for_plot}) - {model_name_prefix}"); plt.tight_layout(); plt.savefig(shap_output_dir / f"{model_name_prefix}_shap_feat_imp.png", bbox_inches='tight'); plt.close()
        
        if X_data_np.shape[0] > 0:
            plt.figure(); base_val_wf = expected_value_for_plotting if not isinstance(expected_value_for_plotting, (list,tuple,np.ndarray)) or len(expected_value_for_plotting)==0 else expected_value_for_plotting[0] if hasattr(expected_value_for_plotting, '__len__') else 0.0
            try: 
                shap.waterfall_plot(shap.Explanation(values=shap_values_for_plotting[0], base_values=base_val_wf, data=X_data_np[0], feature_names=feature_names), max_display=15, show=False); plt.title(f"SHAP Waterfall (Inst 0, {class_name_for_plot}) - {model_name_prefix}"); plt.tight_layout(); plt.savefig(shap_output_dir / f"{model_name_prefix}_shap_waterfall_inst_0.png", bbox_inches='tight')
            except Exception as e: logger.error(f"Waterfall plot failed for {model_name_prefix}: {e}")
            finally: plt.close()
    else: # Multiclass
        if not isinstance(explainer_shap_values_list, list) or not explainer_shap_values_list: logger.error(f"Expected list for multiclass {model_name_prefix}"); return None,None
        indices_to_process = range(len(explainer_shap_values_list)) if class_indices_to_explain is None else [class_indices_to_explain] if isinstance(class_indices_to_explain, int) else class_indices_to_explain
        actual_num_classes_shap = len(explainer_shap_values_list)
        for class_idx in indices_to_process:
            if not (0 <= class_idx < actual_num_classes_shap): logger.warning(f"Class index {class_idx} out of bounds for {model_name_prefix}"); continue
            shap_values_for_class = explainer_shap_values_list[class_idx]
            if np.any(np.abs(shap_values_for_class) > 1e5): logger.warning(f"Extreme SHAP values for {model_name_prefix} Class {class_idx} (max abs: {np.max(np.abs(shap_values_for_class)):.2e})")
            expected_value_for_class = explainer.expected_value[class_idx] if isinstance(explainer.expected_value, (list,np.ndarray)) and len(explainer.expected_value) > class_idx else explainer.expected_value
            class_name = f"Class_{label_encoder_for_class_names.inverse_transform([class_idx])[0]}" if label_encoder_for_class_names and class_idx < len(label_encoder_for_class_names.classes_) else f"Class_{class_idx}"
            
            plt.figure()
            shap.summary_plot(shap_values_for_class, X_data_np, feature_names=feature_names, show=False, plot_type="dot"); plt.title(f"SHAP Summary ({class_name}) - {model_name_prefix}"); plt.tight_layout(); plt.savefig(shap_output_dir / f"{model_name_prefix}_{class_name}_shap_summary.png", bbox_inches='tight'); plt.close()
            
            plt.figure()
            shap.summary_plot(shap_values_for_class, X_data_np, feature_names=feature_names, show=False, plot_type="bar"); plt.title(f"SHAP Bar ({class_name}) - {model_name_prefix}"); plt.tight_layout(); plt.savefig(shap_output_dir / f"{model_name_prefix}_{class_name}_shap_feat_imp.png", bbox_inches='tight'); plt.close()
            if X_data_np.shape[0] > 0:
                plt.figure(); base_val_wf_mc = expected_value_for_class[0] if hasattr(expected_value_for_class, '__len__') and len(expected_value_for_class)>0 else expected_value_for_class if not hasattr(expected_value_for_class, '__len__') else 0.0
                try: 
                    shap.waterfall_plot(shap.Explanation(values=shap_values_for_class[0], base_values=base_val_wf_mc, data=X_data_np[0], feature_names=feature_names), max_display=15, show=False); plt.title(f"SHAP Waterfall (Inst 0, {class_name}) - {model_name_prefix}"); plt.tight_layout(); plt.savefig(shap_output_dir / f"{model_name_prefix}_{class_name}_shap_waterfall_inst_0.png", bbox_inches='tight')
                    # inst 1, class 1
                    shap.waterfall_plot(shap.Explanation(values=shap_values_for_class[1], base_values=base_val_wf_mc, data=X_data_np[1], feature_names=feature_names), max_display=15, show=False); plt.title(f"SHAP Waterfall (Inst 1, class1) - {model_name_prefix}"); plt.tight_layout(); plt.savefig(shap_output_dir / f"{model_name_prefix}_class1_shap_waterfall_inst_1.png", bbox_inches='tight')
                except Exception as e: logger.error(f"Waterfall plot failed for {model_name_prefix} class {class_name}: {e}")
                finally: plt.close()
    
    # --- PLOTTING LOGIC ---

    # Create a list to hold all explanation objects we need to plot
    explanations_to_plot = []
    class_names = []

    # Case 1: Standard Multiclass or Binary output (list of 2+ arrays)
    if isinstance(explainer_shap_values_list, list) and len(explainer_shap_values_list) >= 2:
        logger.info("SHAP returned a list of arrays. Processing each class.")
        for i, values in enumerate(explainer_shap_values_list):
            explanations_to_plot.append(shap.Explanation(
                values=values, base_values=explainer.expected_value[i],
                data=X_data_np, feature_names=feature_names
            ))
            class_names.append(f"Class {i}")

    # Case 2: Special Binary output (list with a single array)
    elif isinstance(explainer_shap_values_list, list) and len(explainer_shap_values_list) == 1:
        logger.info("SHAP returned a list with one array for binary case. Creating plots for both classes.")
        shap_values_c1 = explainer_shap_values_list[0]
        base_value_c1 = explainer.expected_value[0]
        
        # Add explanation for Class 1
        explanations_to_plot.append(shap.Explanation(
            values=shap_values_c1, base_values=base_value_c1,
            data=X_data_np, feature_names=feature_names))
        class_names.append("Class 1")

        # Manually create and add explanation for Class 0
        explanations_to_plot.append(shap.Explanation(
            values=-shap_values_c1, base_values=(1.0 - base_value_c1),
            data=X_data_np, feature_names=feature_names))
        class_names.append("Class 0")

    # Case 3: Special Binary output (single NumPy array)
    elif isinstance(explainer_shap_values_list, np.ndarray):
        logger.info("SHAP returned a single NumPy array for binary case. Creating plots for both classes.")
        shap_values_c1 = explainer_shap_values_list
        base_value_c1 = explainer.expected_value

        # Add explanation for Class 1
        explanations_to_plot.append(shap.Explanation(
            values=shap_values_c1, base_values=base_value_c1,
            data=X_data_np, feature_names=feature_names))
        class_names.append("Class 1")
        
        # Manually create and add explanation for Class 0
        explanations_to_plot.append(shap.Explanation(
            values=-shap_values_c1, base_values=(1.0 - base_value_c1),
            data=X_data_np, feature_names=feature_names))
        class_names.append("Class 0")
        
    # --- Generate all plots by looping through the explanation(s) ---
    if not explanations_to_plot:
        logger.error("Could not create any SHAP explanation objects. Aborting plotting.")
        return explainer_shap_values_list, explainer
        
    for i, expl in enumerate(explanations_to_plot):
        class_name_for_file = class_names[i].replace(' ', '_')

        # --- Instance-wise Waterfall Plots ---
        instances_to_plot = [0, 1, 2, 3, 4, 9]
        for inst_idx in instances_to_plot:
            if inst_idx >= expl.values.shape[0]: continue
            
            instance_explanation = expl[inst_idx]
            
            # 1. Standard Waterfall Plot
            # plt.figure(); 
            # shap.plots.waterfall(instance_explanation, max_display=15, show=False)
            # plt.title(f"SHAP Waterfall (Inst {inst_idx}, {class_names[i]}) - {model_name_prefix}")
            # plt.tight_layout()
            # plt.savefig(shap_output_dir / f"{model_name_prefix}_waterfall_inst_{inst_idx}_{class_name_for_file}.png")
            # plt.close()
            try:
                # --- 1. Get Logits for BOTH Classes ---
                # `explanations_to_plot` is ordered [Class Positive, Class Negative]
                expl_c1 = explanations_to_plot[0][inst_idx]
                expl_c0 = explanations_to_plot[1][inst_idx]
                logit_c1 = expl_c1.base_values + expl_c1.values.sum()
                logit_c0 = expl_c0.base_values + expl_c0.values.sum()

                # --- 2. Apply Softmax for True Probabilities ---
                logits = np.array([logit_c0, logit_c1])
                softmax_probs = softmax(logits) # Stable softmax from scipy.special
                prob_c0, prob_c1 = softmax_probs[0], softmax_probs[1]

                # --- 3. Generate Plot for Class 1 ---
                title_c1 = (f"SHAP Waterfall (Logit Space) - Inst {inst_idx}, Class 1\n"
                            f"{model_name_prefix} | Certainty: {prob_c1:.1%}")
                filename_c1 = f"{model_name_prefix}_waterfall_LOGIT_inst_{inst_idx}_Class_1.png"

                plt.figure()
                shap.plots.waterfall(expl_c1, max_display=10, show=False)
                plt.title(title_c1)
                plt.tight_layout()
                plt.savefig(shap_output_dir / filename_c1)
                plt.close()

                # --- 4. Generate Plot for Class 0 ---
                title_c0 = (f"SHAP Waterfall (Logit Space) - Inst {inst_idx}, Class 0\n"
                            f"{model_name_prefix} | Certainty: {prob_c0:.1%}")
                filename_c0 = f"{model_name_prefix}_waterfall_LOGIT_inst_{inst_idx}_Class_0.png"

                plt.figure()
                shap.plots.waterfall(expl_c0, max_display=10, show=False)
                plt.title(title_c0)
                plt.tight_layout()
                plt.savefig(shap_output_dir / filename_c0)
                plt.close()

            except Exception as e:
                logger.error(f"Could not generate waterfall plots for instance {inst_idx}: {e}")
            
            # 2. Grouped Waterfall Plot
            if group_metrics_list and num_zones_per_metric and (len(group_metrics_list) * num_zones_per_metric == len(feature_names)):
                try:
                    # Sum the SHAP values for each group of features
                    grouped_vals = [instance_explanation.values[j*num_zones_per_metric:(j+1)*num_zones_per_metric].sum() for j in range(len(group_metrics_list))]
                    
                    # Create a new explanation with the grouped values
                    grouped_expl = shap.Explanation(
                        values=np.array(grouped_vals), 
                        base_values=instance_explanation.base_values, 
                        feature_names=group_metrics_list
                    )
                    
                    plt.figure(); shap.plots.waterfall(grouped_expl, max_display=20, show=False)
                    plt.title(f"Grouped SHAP Waterfall (Inst {inst_idx}, {class_names[i]}) - {model_name_prefix}")
                    plt.tight_layout(); plt.savefig(shap_output_dir / f"{model_name_prefix}_grouped_waterfall_inst_{inst_idx}_{class_name_for_file}.png"); plt.close()
                except Exception as e:
                    logger.error(f"Could not generate grouped waterfall for instance {inst_idx}: {e}")
                    
        # --- DEBUGGED CIRCULAR PLOT LOGIC ---
        logger.info("--- Checking conditions for circular plots... ---")
        logger.info(f"Group metrics available: {bool(group_metrics_list)} (count: {len(group_metrics_list) if group_metrics_list else 0})")
        logger.info(f"Num zones per metric: {num_zones_per_metric}")
        logger.info(f"Feature names count: {len(feature_names) if feature_names else 0}")
        if group_metrics_list:
             logger.info(f"Expected feature count: {len(group_metrics_list) * (num_zones_per_metric or 0)}")

        if group_metrics_list and num_zones_per_metric and (len(group_metrics_list) * num_zones_per_metric == len(feature_names)):
            logger.info("Conditions MET. Generating circular plots.")
            # Global Circular Plot
            plot_circular_shap_heatmap(
                shap_values_array=np.abs(expl.values).mean(axis=0),
                group_metrics=group_metrics_list, num_zones_per_metric=num_zones_per_metric,
                model_name_prefix=model_name_prefix, shap_output_dir=shap_output_dir,
                plot_title=f"Global Mean(|SHAP|) - {model_name_prefix} ({class_names[i]})",
                filename_suffix=f"global_{class_name_for_file}", is_global=True
            )

            # Instance-wise Circular Plots
            for inst_idx in instances_to_plot:
                if inst_idx < expl.values.shape[0]:
                    plot_circular_shap_heatmap(
                        shap_values_array=expl.values[inst_idx],
                        group_metrics=group_metrics_list, num_zones_per_metric=num_zones_per_metric,
                        model_name_prefix=model_name_prefix, shap_output_dir=shap_output_dir,
                        plot_title=f"Instance SHAP - {model_name_prefix} (Sample {inst_idx}, {class_names[i]})",
                        filename_suffix=f"instance_{inst_idx}_{class_name_for_file}", is_global=False
                    )
        else:
            logger.error("--- Conditions for circular plots were NOT MET. Skipping. Please check the logs above. ---")
        # --- END         
    
    
    # --- Grouped SHAP Heatmap (Clustermap) ---
    if group_metrics_list and num_zones_per_metric and num_zones_per_metric > 0 and \
       feature_names and len(feature_names) == (len(group_metrics_list) * num_zones_per_metric):
        
        shap_val_for_grouped_plot = None # Renamed to avoid conflict
        if is_binary_mode_flag:
            shap_val_for_grouped_plot = shap_values_for_plotting
        elif isinstance(explainer_shap_values_list, list) and explainer_shap_values_list:
            shap_val_for_grouped_plot = explainer_shap_values_list[0] 
            logger.info(f"For multiclass grouped SHAP plot for {model_name_prefix}, using SHAP values for class 0.")
            
        heatmap_data = pd.DataFrame() 
        
        if shap_val_for_grouped_plot is not None:
            if np.any(np.abs(shap_val_for_grouped_plot) > 1e5): logger.warning(f"Grouped SHAP plot for {model_name_prefix} may be affected by extreme SHAP values.")

            try:
                num_metrics = len(group_metrics_list)
                if shap_val_for_grouped_plot.shape[1] == len(feature_names):
                    mean_abs_shap_reshaped = np.mean(np.abs(shap_val_for_grouped_plot), axis=0).reshape(num_metrics, num_zones_per_metric)
                    # df_mean_abs_shap becomes heatmap_data for the clustermap call
                    heatmap_data = pd.DataFrame(mean_abs_shap_reshaped, index=group_metrics_list,
                                                columns=[f"Z{z+1}" for z in range(num_zones_per_metric)])

                    if heatmap_data.shape[0] > 1 and heatmap_data.shape[1] > 1:
                        logger.info(f"Generating SHAP clustermap for {model_name_prefix} with shape {heatmap_data.shape}")
                        hm_cluster = sns.clustermap(heatmap_data, 
                                                    annot=True, fmt=".3f", cmap="viridis",
                                                    # yticklabels=group_metrics_list, # Already set by DataFrame index
                                                    # xticklabels=[f"Z{z+1}" for z in range(num_zones_per_metric)], # Already set by DataFrame columns
                                                    figsize=(max(8, num_zones_per_metric * 0.6), max(6, len(group_metrics_list) * 0.5)),
                                                    linewidths=.5 # Added for aesthetics
                                                   )
                        # Seaborn's clustermap sets yticklabels and xticklabels from the DataFrame index/columns by default if they are named.to override them, ensure the DataFrame doesn't have named index/columns or set them explicitly in clustermap.
                        # The DataFrame already has these as index and columns, so explicit passing might be redundant but harmless.
                        
                        # Adjust labels if needed (clustermap has its own axes)
                        plt.setp(hm_cluster.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
                        plt.setp(hm_cluster.ax_heatmap.get_yticklabels(), rotation=0)
                        # hm_cluster.ax_heatmap.set_xlabel("Zone") # Clustermap handles this differently
                        # hm_cluster.ax_heatmap.set_ylabel("Metric Group")
                        
                        # Title for clustermap
                        title_str = f"Mean Abs SHAP ({'Positive Class' if is_binary_mode_flag else 'Class 0'}) - {model_name_prefix}"
                        hm_cluster.fig.suptitle(title_str, y=1.03, fontsize=10)

                        plt.savefig(shap_output_dir / f"{model_name_prefix}_shap_clustermap.png", bbox_inches='tight')
                        plt.close(hm_cluster.fig)
                    else: 
                        logger.info(f"DataFrame too small for clustermap ({heatmap_data.shape}), using regular heatmap for {model_name_prefix}.")
                        # Fallback to simple heatmap (Optional)
                        heatmap_width = max(8, min(15, num_zones_per_metric * 0.75))
                        heatmap_height = max(5, num_metrics * 0.6)
                        plt.figure(figsize=(heatmap_width, heatmap_height))
                        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="viridis",
                                    yticklabels=group_metrics_list,
                                    xticklabels=[f"Z{z+1}" for z in range(num_zones_per_metric)])
                        plt.title(f"Mean Abs SHAP ({'Positive Class' if is_binary_mode_flag else 'Class 0'}) - {model_name_prefix}", fontsize=10)
                        plt.ylabel("Metric Group")
                        plt.xlabel("Zone")
                        plt.xticks(rotation=45, ha="right")
                        plt.yticks(rotation=0)
                        plt.tight_layout()
                        plt.savefig(shap_output_dir / f"{model_name_prefix}_shap_grouped_heatmap_simple.png", bbox_inches='tight')
                        plt.close()
                else:
                    logger.warning(f"SHAP values shape ({shap_val_for_grouped_plot.shape[1]}) does not match expected features ({len(feature_names)}) for grouped plot of {model_name_prefix}.")
            except Exception as e_group:
                logger.error(f"Failed to generate grouped SHAP plot for {model_name_prefix}: {e_group}")
                logger.error(traceback.format_exc())
        else:
            logger.warning(f"SHAP values for grouped plot were not available for {model_name_prefix}.")
            
        
        if shap_val_for_grouped_plot is not None and group_metrics_list and num_zones_per_metric > 0:
            logger.info(f"Generating SHAP bar plot grouped by original metric for {model_name_prefix}")
            try:
                num_metrics = len(group_metrics_list)
                expected_total_features = num_metrics * num_zones_per_metric

                # Crucial check: Ensure the SHAP values array matches the expected structure
                if shap_val_for_grouped_plot.shape[1] != expected_total_features:
                    logger.error(f"SHAP values feature dimension ({shap_val_for_grouped_plot.shape[1]}) "
                                f"does not match expected total features ({expected_total_features}). "
                                "Cannot create grouped bar plot. Ensure 'shap_val_for_grouped_plot' "
                                "contains SHAP values for all metric-zone combinations and "
                                "'group_metrics_list' and 'num_zones_per_metric' are correct.")
                else:
                    # 1. Calculate mean absolute SHAP values for all fine-grained features (metric_zone)
                    #    shape: (total_features_including_zones,)
                    mean_abs_shaps_flat = np.abs(shap_val_for_grouped_plot).mean(axis=0)

                    # 2. Reshape to (num_metrics, num_zones_per_metric)
                    #    Rows correspond to group_metrics_list, columns to zones.
                    mean_abs_shaps_reshaped = mean_abs_shaps_flat.reshape(num_metrics, num_zones_per_metric)

                    # 3. Aggregate SHAP values for each original metric group.
                    #    Option A: Sum of mean absolute SHAP values of zones (total impact)
                    metric_aggregated_importance = np.sum(mean_abs_shaps_reshaped, axis=1)
                    y_axis_label = "Sum of Mean(|SHAP value|) across Zones"
                    
                    #    Option B: Mean of mean absolute SHAP values of zones (average impact per zone)
                    #    If your image's "Mean |Importance|" implies this, uncomment the next two lines
                    #    and comment out the two lines above for Option A.
                    # metric_aggregated_importance = np.mean(mean_abs_shaps_reshaped, axis=1)
                    # y_axis_label = "Mean of Mean(|SHAP value|) across Zones"


                    # Sort metrics by their aggregated importance for plotting
                    sorted_indices = np.argsort(metric_aggregated_importance)[::-1] # Descending
                    sorted_metric_names = np.array(group_metrics_list)[sorted_indices]
                    sorted_metric_importance = metric_aggregated_importance[sorted_indices]

                    # 4. Create the vertical bar plot
                    #    Figure size: width, height. Adjust as needed.
                    fig_width = max(8, num_metrics * 0.8) # Adjust width based on number of metrics
                    fig_grouped_bar, ax_grouped_bar = plt.subplots(figsize=(fig_width, 6))

                    bars = ax_grouped_bar.bar(
                        sorted_metric_names,
                        sorted_metric_importance
                    )
                    
                    # Add SHAP values on top of each bar ---
                    for bar in bars:
                        yval = bar.get_height()
                        ax_grouped_bar.text(
                            bar.get_x() + bar.get_width()/2.0,
                            yval,
                            f'{yval:.4f}', # Format 
                            va='bottom', # Vertically align to the bottom of the text
                            ha='center', # Horizontally align to the center
                            fontsize=10
                        )

                    ax_grouped_bar.set_ylabel(y_axis_label) # Set based on aggregation choice
                    # ax_grouped_bar.set_xlabel("Original Metric Group") # X-axis now has metric names
                    plt.xticks(rotation=45, ha="right") # Rotate metric names if they overlap

                    plot_title_detail = 'Positive Class' if is_binary_mode_flag else 'Class 0' # Or use class name if available
                    ax_grouped_bar.set_title(f"Feature Group Importance - {model_name_prefix} ({plot_title_detail})", fontsize=12)
                    ax_grouped_bar.set_ylim(top=ax_grouped_bar.get_ylim()[1] * 1.1)
                    fig_grouped_bar.tight_layout(pad=1.0) # Adjust layout

                    grouped_bar_plot_filename = shap_output_dir / f"{model_name_prefix}_shap_feature_group_importance_plot.png"
                    fig_grouped_bar.savefig(grouped_bar_plot_filename, dpi=150) # bbox_inches='tight' might also be useful
                    plt.close(fig_grouped_bar)
                    logger.info(f"Saved feature group importance bar plot to: {grouped_bar_plot_filename}")

            except Exception as e:
                logger.error(f"Error generating metric group importance bar plot for {model_name_prefix}: {e}", exc_info=True)


            
         # --- Circular Heatmap Plotting Logic --- using the `heatmap_data` above.
        if not heatmap_data.empty:
            logger.info(f"Generating circular SHAP heatmaps for {model_name_prefix}")

            # Define radii for the rings
            r_inner = 0.3 
            r_middle = 0.6
            r_outer = 1.0


            new_zone_definitions = []
            
            start_segment_offsets = {
                    "inner": 0,  # Label 1 starts at geometric segment 0 (NE)
                    "middle": 1, # Label 5 starts at geometric segment 1 (the one just CW from NE)
                    "outer": 1   # Label 13 starts at geometric segment 1
                }

            # Innermost Ring (Target Display Labels: 1 (NE), 2 (NW), 3 (SW), 4 (SE))
            num_segments_inner = 4
            angles_inner_vad = np.linspace(0, 360, num_segments_inner + 1) # VAD: [0, 90, 180, 270, 360]
            base_label_inner = 1
            target_start_idx_inner = start_segment_offsets["inner"]
            for i in range(num_segments_inner): # i is the geometric segment index (0 to N-1), CW from North
                k_raw = target_start_idx_inner - i
                # k is the label sequence index (0 for base_label, 1 for base_label+1, etc. in CCW order)
                k = (k_raw % num_segments_inner + num_segments_inner) % num_segments_inner
                actual_display_label_val = base_label_inner + k
                
                new_zone_definitions.append({
                    "data_col_name": f"Z{actual_display_label_val}",
                    "display_label": str(actual_display_label_val),
                    "r": r_inner,
                    "theta1": angles_inner_vad[i],
                    "theta2": angles_inner_vad[i+1],
                    "annulus_width": r_inner
                })

            # Middle Ring (Target Display Labels: 5 (NE-most), 6 (next CCW), ..., 12)
            num_segments_middle = 8
            angles_middle_vad = np.linspace(0, 360, num_segments_middle + 1)
            base_label_middle = 5
            target_start_idx_middle = start_segment_offsets["middle"]
            for i in range(num_segments_middle):
                k_raw = target_start_idx_middle - i
                k = (k_raw % num_segments_middle + num_segments_middle) % num_segments_middle
                actual_display_label_val = base_label_middle + k
                
                new_zone_definitions.append({
                    "data_col_name": f"Z{actual_display_label_val}",
                    "display_label": str(actual_display_label_val),
                    "r": r_middle,
                    "theta1": angles_middle_vad[i],
                    "theta2": angles_middle_vad[i+1],
                    "annulus_width": r_middle - r_inner
                })

            # Outer Ring
            num_segments_outer = 8
            angles_outer_vad = np.linspace(0, 360, num_segments_outer + 1)
            base_label_outer = 13
            target_start_idx_outer = start_segment_offsets["outer"]
            for i in range(num_segments_outer):
                k_raw = target_start_idx_outer - i
                k = (k_raw % num_segments_outer + num_segments_outer) % num_segments_outer
                actual_display_label_val = base_label_outer + k
                    
                new_zone_definitions.append({
                    "data_col_name": f"Z{actual_display_label_val}",
                    "display_label": str(actual_display_label_val),
                    "r": r_outer,
                    "theta1": angles_outer_vad[i],
                    "theta2": angles_outer_vad[i+1],
                    "annulus_width": r_outer - r_middle
                })
            zone_definitions = new_zone_definitions

###########  generate composite image with all 9 feat HM circular plots ---

            num_features = len(heatmap_data.index)
            if num_features == 0:
                logger.info("No features to plot in composite image.")
            #else: # Proceed to plot if there are features

            # Determine grid size (defaulting to 3x3 for up to 9 features)
            ncols = 3
            nrows = int(np.ceil(num_features / ncols))
            if num_features == 0: # handle case of no features
                logger.info("Skipping composite plot as there are no features.")
            else:
                # Adjust figsize: Aim for each subplot to be reasonably clear.
                # If each subplot needs roughly 4-5 inches, a 3x3 grid could be 12-15 inches wide.
                # Height needs to accommodate suptitle.
                master_fig_width = 4.5 * ncols 
                master_fig_height = 4.5 * nrows + 0.7 # Add a bit more for suptitle

                master_fig, master_axes = plt.subplots(nrows, ncols, figsize=(master_fig_width, master_fig_height))
                
                # If nrows=1 and ncols=1, master_axes is not an array but a single Axes.
                # If nrows=1 or ncols=1 (but not both 1), master_axes is a 1D array.
                # Otherwise, it's a 2D array. Flatten for consistent iteration.
                if num_features == 1:
                    master_axes_flat = [master_axes] # Make it iterable
                else:
                    master_axes_flat = master_axes.flatten()

                # Global normalization for a shared colorbar (already done for individual plots, reuse here)
                # Assuming g_vmin, g_vmax, norm, cmap, sm are already defined globally from single plot logic
                # If not, recalculate them here:
                g_vmin = heatmap_data.min().min()
                g_vmax = heatmap_data.max().max()
                if g_vmin == g_vmax:
                    if g_vmin == 0: g_vmin, g_vmax = -0.001, 0.001
                    else: 
                        abs_g_vmin = abs(g_vmin) # Calculate absolute value once
                        g_vmin = g_vmin - abs_g_vmin * 0.1 if g_vmin != 0 else -0.001
                        g_vmax = g_vmax + abs_g_vmin * 0.1 if g_vmax != 0 else 0.001 # Use abs_g_vmin here too for consistency
                    if g_vmin == g_vmax: g_vmin -=0.001; g_vmax +=0.001 
                            
                current_norm = plt.Normalize(vmin=g_vmin, vmax=g_vmax) # Use current_norm for clarity
                current_cmap = plt.cm.viridis # Or your chosen cmap
                sm = plt.cm.ScalarMappable(cmap=current_cmap, norm=current_norm)
                sm.set_array([])

                # Loop through each feature and plot on a subplot
                for feature_idx, feature_name in enumerate(heatmap_data.index):
                    if feature_idx >= len(master_axes_flat):
                        logger.warning(f"Feature index {feature_idx} exceeds available subplots. Skipping.")
                        break
                        
                    ax = master_axes_flat[feature_idx] # Get the current subplot axes

                    # --- Core Cartesian Drawing Logic for one circular plot ---
                    ax.set_aspect('equal', adjustable='box')
                    current_feature_shap_values = heatmap_data.loc[feature_name]

                    for zone_def in zone_definitions: # Use the reordered zone_definitions
                        shap_value = current_feature_shap_values[zone_def["data_col_name"]]
                        color = current_cmap(current_norm(shap_value)) # Use global norm and cmap

                        r_outer_segment = zone_def["r"]
                        r_inner_segment = zone_def["r"] - zone_def["annulus_width"]
                        vad_start = zone_def["theta1"]
                        vad_end = zone_def["theta2"]
                        wedge_theta1_deg = (90 - vad_end + 360) % 360
                        wedge_theta2_deg = (90 - vad_start + 360) % 360

                        wedge_shape = patches.Wedge(
                            center=(0, 0), r=r_outer_segment,
                            theta1=wedge_theta1_deg, theta2=wedge_theta2_deg,
                            width=r_outer_segment - r_inner_segment
                        )
                        path_patch = patches.PathPatch(
                            wedge_shape.get_path(), facecolor=color,
                            edgecolor='black', linewidth=0.3 # Thinner lines for subplots
                        )
                        ax.add_patch(path_patch)

                        label_vad_center = (vad_start + vad_end) / 2
                        label_radius = r_outer_segment - (r_outer_segment - r_inner_segment) / 2
                        if r_inner_segment == 0: label_radius = r_outer_segment / 1.6
                        label_mad_rad = np.deg2rad((90 - label_vad_center + 360) % 360)
                        x_label = label_radius * np.cos(label_mad_rad)
                        y_label = label_radius * np.sin(label_mad_rad)
                        ax.text(x_label, y_label, zone_def["display_label"],
                                ha='center', va='center', fontsize=5, # Smaller font for subplots
                                color="white" if current_norm(shap_value) < 0.3 else "black")

                    plot_radius_padding = 0.05
                    ax.set_xlim(-r_outer - plot_radius_padding, r_outer + plot_radius_padding)
                    ax.set_ylim(-r_outer - plot_radius_padding, r_outer + plot_radius_padding)
                    ax.axis('off')
                    ax.set_title(feature_name, fontsize=9) # Title for each subplot
                    # --- End of Core Cartesian Drawing Logic ---

                # Turn off any unused subplots if num_features isn't a perfect multiple of nrows*ncols
                for i in range(num_features, nrows * ncols):
                    if i < len(master_axes_flat): # Check index bounds
                        master_axes_flat[i].axis('off')

                # Add a single, shared colorbar to the master figure
                # Adjust [left, bottom, width, height] as needed for your master_fig_width/height
                # These are fractions of the *figure* dimensions.
                cbar_left = 0.91 # Positioned towards the right
                cbar_bottom = 0.15 
                cbar_width = 0.015 # Thin colorbar
                cbar_height = 0.7 # Cover 70% of figure height vertically
                
                # Check if figure has space for colorbar based on calculated width/height
                if master_fig_width * (1 - cbar_left - cbar_width) < 0.5: # Heuristic: if not enough space
                    logger.warning("Figure might be too narrow for the colorbar at specified position. Adjusting.")
                    cbar_left = 0.88 # Try to pull it left a bit if figure is very narrow

                cbar_ax = master_fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
                master_fig.colorbar(sm, cax=cbar_ax, orientation='vertical', label="Mean Abs SHAP")

                # Add overall title to the master figure
                title_detail_composite = 'Positive Class' if is_binary_mode_flag else 'Class 0'
                master_fig.suptitle(f"All Features - Circular SHAP Heatmaps\n(Mean Abs SHAP for {title_detail_composite} - {model_name_prefix})",
                                    fontsize=14, y=0.98) # Adjust y if suptitle overlaps subplots

                # Adjust layout to prevent overlap, considering manually placed colorbar
                # rect can be [left, bottom, right, top]
                master_fig.tight_layout(rect=[0, 0.03, cbar_left - 0.02, 0.95]) # Leave space for cbar and suptitle

                # Save the master figure
                master_fig_filename = shap_output_dir / f"{model_name_prefix}_shap_feat_imp_circ.png"
                master_fig.savefig(master_fig_filename, dpi=150) # Use a reasonable DPI
                plt.close(master_fig) # Close the master figure
                logger.info(f"Saved composite circular SHAP heatmap to: {master_fig_filename}")

        else:
            logger.info(f"Skipping circular SHAP plot for {model_name_prefix} as heatmap_data was not generated or is empty.")

        
        
    elif group_metrics_list and num_zones_per_metric :
         logger.warning(f"Skipping grouped SHAP plot for {model_name_prefix}: Feature count mismatch.")
         
    else:
        logger.info(f"Skipping all grouped SHAP plots for {model_name_prefix} due to initial condition failure.")

    logger.info(f"SHAP analysis for {model_name_prefix} (potentially) completed.")
    
    
    return explainer_shap_values_list, explainer


# --- 4. GRID SEARCH TUNING --- (Simplified version from draft)
def grid_search_tuning_old(model_class, base_model_params, param_grid,
                       X_train_gs, y_train_gs, # These are full train set for GS internal CV
                       optimizer_base_params, criterion_gs, device_gs,
                       gs_report_dir, gs_subdirs,
                       feature_dim_gs, # This is model_params['feature_dim']
                       has_channel_dim_gs,
                       n_splits_gs=3, epochs_gs=30, batch_size_gs=16): # Reduced for speed

    logger.info(f"Starting grid search for {model_class.__name__}...")
    results = []
    param_combinations = list(ParameterGrid(param_grid))
    logger.info(f"Number of parameter combinations to test: {len(param_combinations)}")


    # Ensure X_train_gs, y_train_gs are numpy for splitting
    X_train_gs_np = X_train_gs.cpu().numpy() if isinstance(X_train_gs, torch.Tensor) else X_train_gs
    y_train_gs_np = y_train_gs.cpu().numpy() if isinstance(y_train_gs, torch.Tensor) else y_train_gs


    for i, params_combo in enumerate(param_combinations):
        current_model_params = base_model_params.copy()
        current_model_params.update(params_combo)
        current_model_params.pop('learning_rate', None) # LR is for optimizer
        current_model_params.pop('weight_decay', None) # WD is for optimizer

        current_optimizer_params = optimizer_base_params.copy()
        if 'learning_rate' in params_combo:
            current_optimizer_params['lr'] = params_combo['learning_rate']
        if 'weight_decay' in params_combo:
            current_optimizer_params['weight_decay'] = params_combo['weight_decay']

        logger.info(f"Grid Search - Combination {i+1}/{len(param_combinations)}: {params_combo}")

        skf_gs = StratifiedKFold(n_splits=n_splits_gs, shuffle=True, random_state=42)
        fold_f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf_gs.split(X_train_gs_np, y_train_gs_np)):
            model_gs = model_class(**current_model_params).to(device_gs)
            model_gs.apply(reset_parameters)
            optimizer_gs = torch.optim.AdamW(model_gs.parameters(), **current_optimizer_params)
            # No scheduler or early stopping in this simplified GS for speed, or very lenient.
            # Early stopping can be added if epochs_gs is large.

            _X_fold_train = torch.tensor(X_train_gs_np[train_idx], dtype=torch.float32)
            _y_fold_train = torch.tensor(y_train_gs_np[train_idx], dtype=torch.float32).unsqueeze(1)
            _X_fold_val = torch.tensor(X_train_gs_np[val_idx], dtype=torch.float32)
            _y_fold_val = torch.tensor(y_train_gs_np[val_idx], dtype=torch.float32).unsqueeze(1)

            if has_channel_dim_gs:
                if len(_X_fold_train.shape) == 2: _X_fold_train = _X_fold_train.unsqueeze(1)
                if len(_X_fold_val.shape) == 2: _X_fold_val = _X_fold_val.unsqueeze(1)

            train_ds_gs = TensorDataset(_X_fold_train.to(device_gs), _y_fold_train.to(device_gs))
            val_ds_gs = TensorDataset(_X_fold_val.to(device_gs), _y_fold_val.to(device_gs))
            train_dl_gs = DataLoader(train_ds_gs, batch_size=batch_size_gs, shuffle=True)
            val_dl_gs = DataLoader(val_ds_gs, batch_size=batch_size_gs, shuffle=False)
            
            # Originally, criterion_gs is for multiclass; added fol for binary
            # if 'pos_weight_value' in params_combo:
            #     pos_weight_tensor_gs = torch.tensor([params_combo['pos_weight_value']], dtype=torch.float32).to(device_gs)
            #     criterion_gs = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor_gs)
            # else:
            #     # Fallback to the original criterion (e.g., for multiclass or if pos_weight not in grid)
            #     pass 

            for epoch in range(epochs_gs):
                model_gs.train()
                for inputs, labels in train_dl_gs:
                    optimizer_gs.zero_grad()
                    outputs = model_gs(inputs)
                    loss = criterion_gs(outputs, labels)
                    loss.backward()
                    optimizer_gs.step()

            # Evaluate on val fold
            model_gs.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_dl_gs:
                    outputs = model_gs(inputs)
                    probs = torch.sigmoid(outputs)
                    preds = (probs >= 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            f1_val = f1_score(np.array(val_labels).flatten(), np.array(val_preds).flatten(), average='binary', zero_division=0)
            fold_f1_scores.append(f1_val)
            logger.debug(f"    Fold {fold+1} Val F1: {f1_val:.4f}")


        avg_f1 = np.mean(fold_f1_scores) if len(fold_f1_scores) > 0 else 0
        std_f1 = np.std(fold_f1_scores) if len(fold_f1_scores) > 0 else 0
        results.append({'params': params_combo, 'avg_f1': avg_f1, 'std_f1': std_f1, 'fold_scores':fold_f1_scores})
        logger.info(f"  Params: {params_combo} -> Avg F1: {avg_f1:.4f} +/- {std_f1:.4f}")

    results_df = pd.DataFrame([{**r['params'], 'avg_f1': r['avg_f1'], 'std_f1': r['std_f1']} for r in results])
    results_df = results_df.sort_values(by='avg_f1', ascending=False)
    results_df.to_csv(gs_subdirs['metrics'] / f"{model_class.__name__}_grid_search_results.csv", index=False)

    best_params_combo = results_df.iloc[0].to_dict() if not results_df.empty else {}
    best_avg_f1_score = best_params_combo.get('avg_f1', -1.0)
    logger.info(f"Best parameters for {model_class.__name__}: {best_params_combo} (Avg F1: {best_params_combo.get('avg_f1',0):.4f})")

    ### ADDED
     # --- Parameters to be returned ---
    # Start with a copy of the original base_model_params (passed into this function)
    # These usually contain non-tunable structural params like feature_dim, num_classes
    # We will update this with the tuned hyperparameters from best_params_combo.
    
    # Tunable MODEL parameters that should be integers
    integer_model_params_keys = ['cnn_units', 'transformer_dim', 'transformer_layers', 'transformer_heads']
    
    # Initialize dictionaries for the final tuned parameters
    final_tuned_model_hyperparams = {} # For hyperparameters like cnn_units, dropout_rate etc.
    final_tuned_optimizer_hyperparams = {} # For lr, weight_decay

    if best_params_combo: # If tuning found any best combination
        for key, value in best_params_combo.items():
            if key in integer_model_params_keys:
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    final_tuned_model_hyperparams[key] = int(value)
            elif key == 'dropout_rate': # Example of a float model hyperparameter
                final_tuned_model_hyperparams[key] = float(value)
            # Add other specific float model hyperparameters here if any from the grid
            
            # Optimizer hyperparameters
            elif key == 'learning_rate':
                final_tuned_optimizer_hyperparams['lr'] = float(value)
            elif key == 'weight_decay':
                final_tuned_optimizer_hyperparams['weight_decay'] = float(value)
    ##

    # OG >>
    # final_best_model_params = base_model_params.copy()
    # model_specific_best_params = {k: v for k,v in best_params_combo.items() if k in final_best_model_params or k in ['cnn_units', 'transformer_dim', 'transformer_heads', 'transformer_layers', 'dropout_rate']}
    # final_best_model_params.update(model_specific_best_params)

    # final_best_optimizer_params = optimizer_base_params.copy()
    # if 'learning_rate' in best_params_combo: final_best_optimizer_params['lr'] = best_params_combo['learning_rate']
    # if 'weight_decay' in best_params_combo: final_best_optimizer_params['weight_decay'] = best_params_combo['weight_decay']

    # Plot grid search results (simplified)
    if len(param_grid) > 0 and not results_df.empty:
        main_param_to_plot = list(param_grid.keys())[0]
        if main_param_to_plot in results_df.columns:
            try:
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=results_df, x=main_param_to_plot, y='avg_f1', marker='o')
                plt.title(f'Grid Search: {main_param_to_plot} vs. Avg F1 ({model_class.__name__})')
                plt.xlabel(main_param_to_plot)
                plt.ylabel('Average F1 Score (on validation folds)')
                plt.grid(True)
                save_figure(plt.gcf(), f"{model_class.__name__}_gs_{main_param_to_plot}_impact", gs_subdirs['figures'])
            except Exception as e:
                logger.error(f"Could not plot GS results for {main_param_to_plot}: {e}")

    # return final_best_model_params, final_best_optimizer_params, best_avg_f1_score
    return final_tuned_model_hyperparams, final_tuned_optimizer_hyperparams, best_avg_f1_score


def grid_search_tuning(model_class, base_model_params, param_grid_list,
                       X_train_gs, y_train_gs, # These are full train set for GS internal CV
                       optimizer_base_params, criterion_gs, device_gs,
                       gs_report_dir, gs_subdirs,
                       feature_dim_gs, # This is model_params['feature_dim']
                       has_channel_dim_gs,
                       n_splits_gs=3, epochs_gs=30, batch_size_gs=16, # Reduced for speed
                       is_binary_mode=False, worker_results_file=None): # New parameter to handle binary/multiclass

    logger.info(f"Starting grid search for {model_class.__name__}...")
    results = []
    # param_combinations = list(ParameterGrid(param_grid))
    param_combinations = param_grid_list
    logger.info(f"Number of parameter combinations to test: {len(param_combinations)}")


    # Ensure X_train_gs, y_train_gs are numpy for splitting
    X_train_gs_np = X_train_gs.cpu().numpy() if isinstance(X_train_gs, torch.Tensor) else X_train_gs
    y_train_gs_np = y_train_gs.cpu().numpy() if isinstance(y_train_gs, torch.Tensor) else y_train_gs


    for i, params_combo in enumerate(param_combinations):
        current_model_params = base_model_params.copy()
        current_model_params.update(params_combo)
        # current_model_params.pop('learning_rate', None) # LR is for optimizer
        # current_model_params.pop('weight_decay', None) # WD is for optimizer

        # --- FIX: POP LR/WD before model instantiation
        current_optimizer_params = optimizer_base_params.copy()
        if 'learning_rate' in current_model_params:
            current_optimizer_params['lr'] = current_model_params.pop('learning_rate')
        if 'weight_decay' in current_model_params:
            current_optimizer_params['weight_decay'] = current_model_params.pop('weight_decay')
        
        # --- FIX: SEPARATE POS_WEIGHT FOR LOSS FUNCTION
        pos_weight_value = None
        if is_binary_mode and 'pos_weight_value' in current_model_params:
            pos_weight_value = current_model_params.pop('pos_weight_value')

        logger.info(f"Grid Search - Combination {i+1}/{len(param_combinations)}: {params_combo}")


        skf_gs = StratifiedKFold(n_splits=n_splits_gs, shuffle=True, random_state=42)
        fold_f1_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf_gs.split(X_train_gs_np, y_train_gs_np)):
            model_gs = model_class(**current_model_params).to(device_gs)
            model_gs.apply(reset_parameters)
            optimizer_gs = torch.optim.AdamW(model_gs.parameters(), **current_optimizer_params)
            # --- FIX: INSTANTIATE LOSS FUNCTION WITH CORRECT POS_WEIGHT
            criterion_gs_current = criterion_gs
            if is_binary_mode and pos_weight_value is not None:
                pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device_gs)
                criterion_gs_current = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
            
            _X_fold_train = torch.tensor(X_train_gs_np[train_idx], dtype=torch.float32)
            _y_fold_train = torch.tensor(y_train_gs_np[train_idx], dtype=torch.float32).unsqueeze(1)
            _X_fold_val = torch.tensor(X_train_gs_np[val_idx], dtype=torch.float32)
            _y_fold_val = torch.tensor(y_train_gs_np[val_idx], dtype=torch.float32).unsqueeze(1)

            if has_channel_dim_gs:
                if len(_X_fold_train.shape) == 2: _X_fold_train = _X_fold_train.unsqueeze(1)
                if len(_X_fold_val.shape) == 2: _X_fold_val = _X_fold_val.unsqueeze(1)

            train_ds_gs = TensorDataset(_X_fold_train.to(device_gs), _y_fold_train.to(device_gs))
            val_ds_gs = TensorDataset(_X_fold_val.to(device_gs), _y_fold_val.to(device_gs))
            train_dl_gs = DataLoader(train_ds_gs, batch_size=batch_size_gs, shuffle=True)
            val_dl_gs = DataLoader(val_ds_gs, batch_size=batch_size_gs, shuffle=False)

            for epoch in range(epochs_gs):
                model_gs.train()
                for inputs, labels in train_dl_gs:
                    optimizer_gs.zero_grad()
                    outputs = model_gs(inputs)
                    # loss = criterion_gs(outputs, labels)
                    loss = criterion_gs_current(outputs, labels) # Updated
                    loss.backward()
                    optimizer_gs.step()

            # Evaluate on val fold
            model_gs.eval()
            val_preds, val_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_dl_gs:
                    outputs = model_gs(inputs)
                    probs = torch.sigmoid(outputs)
                    preds = (probs >= 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            f1_val = f1_score(np.array(val_labels).flatten(), np.array(val_preds).flatten(), average='binary', zero_division=0)
            fold_f1_scores.append(f1_val)
            logger.debug(f"    Fold {fold+1} Val F1: {f1_val:.4f}")


        avg_f1 = np.mean(fold_f1_scores) if len(fold_f1_scores) > 0 else 0
        std_f1 = np.std(fold_f1_scores) if len(fold_f1_scores) > 0 else 0
        results.append({'params': params_combo, 'avg_f1': avg_f1, 'std_f1': std_f1, 'fold_scores':fold_f1_scores})
        logger.info(f"  Params: {params_combo} -> Avg F1: {avg_f1:.4f} +/- {std_f1:.4f}")

    results_df = pd.DataFrame([{**r['params'], 'avg_f1': r['avg_f1'], 'std_f1': r['std_f1']} for r in results])
    results_df = results_df.sort_values(by='avg_f1', ascending=False)
    # results_df.to_csv(gs_subdirs['metrics'] / f"{model_class.__name__}_grid_search_results.csv", index=False)
    # alternative added
    if worker_results_file:
        results_df.to_csv(worker_results_file, index=False)
        logger.info(f"Saved worker results to {worker_results_file}")
    else:
        # Fallback for single-GPU run, keeps original behavior
        results_df.to_csv(gs_subdirs['metrics'] / f"{model_class.__name__}_grid_search_results.csv", index=False)


    best_params_combo = results_df.iloc[0].to_dict() if not results_df.empty else {}
    best_avg_f1_score = best_params_combo.get('avg_f1', -1.0)
    logger.info(f"Best parameters for {model_class.__name__}: {best_params_combo} (Avg F1: {best_params_combo.get('avg_f1',0):.4f})")

    # Updated
    # Define the keys for different parameter types
    # Tunable MODEL parameters that should be integers
    integer_model_params_keys = ['cnn_units', 'transformer_dim', 'transformer_layers', 'transformer_heads']
    float_optimizer_params_keys = ['learning_rate', 'weight_decay']
    float_model_params_keys = ['dropout_rate', 'cnn_dropout', 'transformer_dropout', 'fc_dropout', 'pos_weight_value']

    # Initialize dictionaries for the final tuned parameters
    final_tuned_model_hyperparams = {} # For hyperparameters like cnn_units, dropout_rate etc.
    final_tuned_optimizer_hyperparams = {} # For lr, weight_decay

    if best_params_combo: # If tuning found any best combination
        for key, value in best_params_combo.items():
            if key in integer_model_params_keys:
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    final_tuned_model_hyperparams[key] = int(value)
            # elif key == 'dropout_rate': # Example of a float model hyperparameter
            #     final_tuned_model_hyperparams[key] = float(value)
            # # Add other specific float model hyperparameters here if any from the grid
            # # Optimizer hyperparameters
            # elif key == 'learning_rate':
            #     final_tuned_optimizer_hyperparams['lr'] = float(value)
            # elif key == 'weight_decay':
            #     final_tuned_optimizer_hyperparams['weight_decay'] = float(value)
            elif key in float_model_params_keys:
                if value is not None and not (isinstance(value, float) and math.isnan(value)):
                    final_tuned_model_hyperparams[key] = float(value)
            elif key in float_optimizer_params_keys:
                if key == 'learning_rate':
                    final_tuned_optimizer_hyperparams['lr'] = float(value)
                elif key == 'weight_decay':
                    final_tuned_optimizer_hyperparams['weight_decay'] = float(value)
    ##    
    # Add pos_weight_value to the model params dict
    # if 'pos_weight_value' in best_params_combo:
    #     final_tuned_model_hyperparams['pos_weight_value'] = best_params_combo['pos_weight_value']


    # OG >>
    # final_best_model_params = base_model_params.copy()
    # model_specific_best_params = {k: v for k,v in best_params_combo.items() if k in final_best_model_params or k in ['cnn_units', 'transformer_dim', 'transformer_heads', 'transformer_layers', 'dropout_rate']}
    # final_best_model_params.update(model_specific_best_params)

    # final_best_optimizer_params = optimizer_base_params.copy()
    # if 'learning_rate' in best_params_combo: final_best_optimizer_params['lr'] = best_params_combo['learning_rate']
    # if 'weight_decay' in best_params_combo: final_best_optimizer_params['weight_decay'] = best_params_combo['weight_decay']

    # Plot grid search results (old)
    # if len(param_grid) > 0 and not results_df.empty:
    #     main_param_to_plot = list(param_grid.keys())[0]
    if len(param_grid_list) > 0 and not results_df.empty:
        main_param_to_plot = list(param_grid_list[0].keys())[0] # Get the keys from the first dictionary in the list
        if main_param_to_plot in results_df.columns:
            try:
                plt.figure(figsize=(10, 6))
                sns.lineplot(data=results_df, x=main_param_to_plot, y='avg_f1', marker='o')
                plt.title(f'Grid Search: {main_param_to_plot} vs. Avg F1 ({model_class.__name__})')
                plt.xlabel(main_param_to_plot)
                plt.ylabel('Average F1 Score (on validation folds)')
                plt.grid(True)
                save_figure(plt.gcf(), f"{model_class.__name__}_gs_{main_param_to_plot}_impact", gs_subdirs['figures'])
            except Exception as e:
                logger.error(f"Could not plot GS results for {main_param_to_plot}: {e}")

    # return final_best_model_params, final_best_optimizer_params, best_avg_f1_score
    return final_tuned_model_hyperparams, final_tuned_optimizer_hyperparams, best_avg_f1_score


def grid_search_worker_wrapper(
    device_id, combinations_chunk, model_type, base_model_params,
    X_train_gs_np, y_train_gs_np,
    optimizer_base_params, criterion, gs_report_dir, gs_subdirs,
    feature_dim_gs, has_channel_dim_gs, n_splits_gs, epochs_gs,
    batch_size_gs, is_binary_mode):
    """
    Wrapper function that runs a chunk of the grid search on a single GPU.
    It saves results to a unique CSV file to avoid conflicts.
    """
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
        device = torch.device(f"cuda:{device_id}")
        logger.info(f"Worker for device {device} starting. Processing {len(combinations_chunk)} combinations.")

        # --- FIX: Instantiate model_class_ref inside the worker ---
        # It uses the `model_type` string to define the model class reference
        if model_type == 'CNN':
            model_class_ref = CNNModel
        elif model_type == 'Transformer':
            model_class_ref = TransformerModel
        elif model_type in ['CNNTransformer_sequential', 'CNNTransformer_parallel']:
            model_class_ref = CNNTransformerModel
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # --- FIX: Instantiate criterion inside the worker ---
        if is_binary_mode:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        # Create a unique file for this worker's results
        # worker_results_path = gs_subdirs['metrics'] / f"{model_class.__name__}_grid_search_worker_{device_id}.csv"
        # Note: 'model_type' is now used directly here as it is a string
        worker_results_path = gs_subdirs['metrics'] / f"{model_type}_grid_search_worker_{device_id}.csv"

        # Call the existing grid_search_tuning function with the chunk
        tuned_model_hparams, tuned_optim_hparams, best_tuning_score = grid_search_tuning(
            model_class=model_class_ref, #model_class,
            base_model_params=base_model_params,
            param_grid_list=combinations_chunk,  # Pass the list directly
            X_train_gs=X_train_gs_np,
            y_train_gs=y_train_gs_np,
            optimizer_base_params=optimizer_base_params,
            criterion_gs=criterion,
            device_gs=device,
            gs_report_dir=gs_report_dir,
            gs_subdirs=gs_subdirs,
            n_splits_gs=n_splits_gs,
            epochs_gs=epochs_gs,
            batch_size_gs=batch_size_gs,
            feature_dim_gs=feature_dim_gs,
            has_channel_dim_gs=has_channel_dim_gs,
            is_binary_mode=is_binary_mode,
            worker_results_file=worker_results_path  # Pass the unique file path
        )
        logger.info(f"Worker for device {device} finished.")
        
    except Exception as e:
        logger.error(f"Worker on device {device_id} failed: {e}")

# --- 5. DATA PREPARATION ---
def prepare_data_for_dl(data_path, is_binary, dataset_name="mpod", stratify_split=True, test_size=0.2, holdout_size=0.15, has_channel_dim_output=True, random_state=42, selected_features=None, test_data_path=None, resampling_method=None):
    ''' Prepare data for deep learning model training and evaluation.
    This function handles three cases: mpod, breast_cancer, and scania_aps'''
    
    # --- Universal CSV Loading ---
    if not data_path:
        raise ValueError("--data_path must be provided.")
    
    # Conditional Feature Handling ---
    if dataset_name == "scania_aps":
        logger.info("Applying feature handling for 'scania_aps' with test file as holdout.")
        if not test_data_path:
            raise ValueError("--test_data_path is required for the scania_aps dataset.")

        # 1. Load data
        df_train_val = pd.read_csv(data_path, na_values="na")
        df_holdout = pd.read_csv(test_data_path, na_values="na")
        logger.info(f"Loaded train_val data: {df_train_val.shape}, Holdout data: {df_holdout.shape}")

        # --- Define X_raw for use in the final dictionary ---
        #
        X_raw = df_train_val.drop('class', axis=1)

        # 2. Process training data
        y_train_val_raw = df_train_val['class']
        X_train_val_raw = df_train_val.drop('class', axis=1) # This is a duplicate of X_raw, which is fine.
        
        le = LabelEncoder()
        y_train_val_encoded = le.fit_transform(y_train_val_raw)
        logger.info(f"LabelEncoder classes: {le.classes_}")

        imputer = SimpleImputer(strategy='median')
        X_train_val_imputed = imputer.fit_transform(X_train_val_raw)
        
        # 3. Process HOLDHOUT data
        y_holdout_raw = df_holdout['class']
        X_holdout_raw = df_holdout.drop('class', axis=1)
        y_holdout_encoded = le.transform(y_holdout_raw)
        X_holdout_imputed = imputer.transform(X_holdout_raw)

        # 4. Perform the internal train/test split on the train_val data
        # The 'X_train_orig' and 'X_test_orig' variables will hold the imputed but unscaled data, as required.
        X_train_orig, X_test_orig, y_train_encoded, y_test_encoded = train_test_split(
            X_train_val_imputed, y_train_val_encoded, test_size=test_size, random_state=random_state, stratify=y_train_val_encoded
        )
        logger.info(f"Split Train_Val: Train set size: {X_train_orig.shape[0]}, Test set size: {X_test_orig.shape[0]}")
        
        # --- FIX: Explicitly define X_holdout_orig ---
        # This variable holds the imputed but unscaled holdout features.
        X_holdout_orig = X_holdout_imputed
        
        # --- NEW RESAMPLING LOGIC ---
        if resampling_method:            
            logger.info(f"Applying resampling method: {resampling_method}")
            if resampling_method == 'SMOTEENN':
                resampler = SMOTEENN(random_state=random_state)
            elif resampling_method == 'SMOTETomek':
                resampler = SMOTETomek(random_state=random_state)
            
            X_train_orig, y_train_encoded = resampler.fit_resample(X_train_orig, y_train_encoded)
            logger.info(f"Resampling complete. New training shape: {X_train_orig.shape}, new target counts: {Counter(y_train_encoded)}")
        # --- END NEW RESAMPLING LOGIC ---


        # 5. Scale all data partitions
        scaler = StandardScaler()
        X_train_scaled_np = scaler.fit_transform(X_train_orig)
        X_test_scaled_np = scaler.transform(X_test_orig)
        X_holdout_scaled_np = scaler.transform(X_holdout_orig) # Scale the correct variable

        # Finalize other variables for the dictionary
        final_feature_names = X_train_val_raw.columns.tolist()
        final_group_metrics = None
        num_features_per_group = 0
        num_classes_found = len(le.classes_)


    else:
        # Performs its own combined train/test/holdout split from a single csv data file.
        logger.info(f"Applying standard split logic. Loading data from {data_path} for dataset '{dataset_name}'")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        # Using all 9 features
        # Adjust if necessary. Target is the last--> df.iloc[:, :-1]
        # num_feature_cols = df.shape[1] - 4 
        # X_raw = df.iloc[:, :num_feature_cols].values.astype(np.float32)
        # y_original_raw = df.iloc[:, -1].values

        if df.shape[1] <= 1:
            raise ValueError("Data must have at least one feature column and one target column.")
            
            
        if dataset_name == "breast_cancer":
            logger.info("Applying simple feature handling for generic tabular dataset with target in the last column.")
            # Assuming target is the last column
            X_raw = df.iloc[:, :-1].values.astype(np.float32)
            y_original_raw = df.iloc[:, -1].values
            
            # Feature names are simply the column headers
            final_feature_names = df.columns[:-1].tolist()
            
            # These are specific to mpod and do not apply here. Set to None/0.
            final_group_metrics = None
            num_features_per_group = 0
            
        elif dataset_name == "mpod":
            logger.info("Applying feature group handling for 'mpod' dataset.")
            
            # Define full feature list first
            # 1. # Define the complete feature group structure
            all_feature_groups = ['mean', 'median', 'std', 'iqr', 'idr', 'skew', 'kurt', 'Del', 'Amp']
            num_features_per_group = 20
            
            # Generate all 180 possible feature names to create a complete reference DataFrame
            all_possible_column_names = [f"{metric}_Z{z+1}" for metric in all_feature_groups for z in range(num_features_per_group)]
            
            # Create a reference DataFrame of just the feature columns with proper names
            feature_df = df.iloc[:, :len(all_possible_column_names)]
            feature_df.columns = all_possible_column_names
        
        
        # Determine which groups and columns to use based on selection
            if selected_features:
                logger.info(f"Selecting feature groups by index: {selected_features}")
                # THIS IS THE CRITICAL FIX: Filter the list of group names
                final_group_metrics = [all_feature_groups[i] for i in selected_features]
                columns_to_keep = []
                for metric_name in final_group_metrics:
                    columns_to_keep.extend([f"{metric_name}_Z{z+1}" for z in range(num_features_per_group)])
            else:
                logger.info("Using all 9 feature groups by default.")
                final_group_metrics = all_feature_groups
                columns_to_keep = all_possible_column_names

            # Select the final data and names
            X_raw = feature_df[columns_to_keep].values.astype(np.float32)
            final_feature_names = columns_to_keep
            
            logger.info(f"Corresponding metric groups: {final_group_metrics}")
            logger.info(f"Final feature dimension for model: {X_raw.shape[1]}")

            y_original_raw = df.iloc[:, -1].values
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")
        # ------
        logger.info(f"Final feature dimension for model: {X_raw.shape[1]}")
        logger.info(f"Original raw target values unique: {np.unique(y_original_raw, return_counts=True)}")

        y_for_le = None
        if is_binary:
            logger.info("Binary classification mode selected. Target will be binarized (0 vs. rest).")
            y_for_le = np.where(y_original_raw == 1, 0, 1).astype(np.int64) # maps original class 1->0, and all others (2,3,4)->1
            logger.info(f"Target after binarization (for LabelEncoder) unique: {np.unique(y_for_le, return_counts=True)}")
        else: # Multiclass mode
            logger.info("Multiclass classification mode selected. Using raw target labels.")
            y_for_le = y_original_raw.astype(np.int64)

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_for_le)
        
        num_classes_found = len(le.classes_)
        logger.info(f"Target encoded. LabelEncoder classes: {le.classes_}, Encoded values range: {np.min(y_encoded)} to {np.max(y_encoded)}")

        if num_classes_found <= 1:
            raise ValueError(f"Only {num_classes_found} class(es) found in target variable after processing: {le.classes_}. Need at least two for classification.")
        if is_binary and num_classes_found > 2:
            logger.warning(f"Binary mode selected, but LabelEncoder found {num_classes_found} classes from the binarized target. "
                        f"This means the binarized target (0s and 1s) was re-encoded by LabelEncoder (e.g., to 0 and 1 if those were the only values). Encoded classes: {le.classes_}")

        # Split 1: Training + Validation vs Holdout
        # These X_ variables are unscaled at this point
        X_train_val_orig, X_holdout_orig, y_train_val_encoded, y_holdout_encoded = (
            np.array([]).reshape(0, X_raw.shape[1]), np.array([]).reshape(0, X_raw.shape[1]),
            np.array([], dtype=y_encoded.dtype), np.array([], dtype=y_encoded.dtype)
        )
        if holdout_size > 0 and holdout_size < 1.0 :
            stratify_holdout = y_encoded if stratify_split else None
            X_train_val_orig, X_holdout_orig, y_train_val_encoded, y_holdout_encoded = train_test_split(
                X_raw, y_encoded, test_size=holdout_size, random_state=random_state, stratify=stratify_holdout
            )
            logger.info(f"Split data: Train_Val set size: {X_train_val_orig.shape[0]}, Holdout set size: {X_holdout_orig.shape[0]}")
            logger.info(f"Holdout class distribution: {Counter(y_holdout_encoded)}")
        else:
            X_train_val_orig, y_train_val_encoded = X_raw, y_encoded
            logger.info("No holdout set created, or holdout_size is invalid.")

        # Split 2: Training vs Test from X_train_val_orig
        X_train_orig, X_test_orig, y_train_encoded, y_test_encoded = (
            np.array([]).reshape(0, X_train_val_orig.shape[1]), np.array([]).reshape(0, X_train_val_orig.shape[1]),
            np.array([], dtype=y_train_val_encoded.dtype), np.array([], dtype=y_train_val_encoded.dtype)
        )
        actual_test_size = test_size
        if X_train_val_orig.shape[0] > 1 and actual_test_size > 0 and actual_test_size < 1.0:
            stratify_train_test = y_train_val_encoded if stratify_split else None
            X_train_orig, X_test_orig, y_train_encoded, y_test_encoded = train_test_split(
                X_train_val_orig, y_train_val_encoded, test_size=actual_test_size, random_state=random_state, stratify=stratify_train_test
            )
            logger.info(f"Split Train_Val: Train set size: {X_train_orig.shape[0]}, Test set size: {X_test_orig.shape[0]}")
        else:
            X_train_orig, y_train_encoded = X_train_val_orig, y_train_val_encoded
            logger.info(f"Using all Train_Val data for training (size: {X_train_orig.shape[0]}). No separate internal test set from this split, or test_size is invalid.")

        logger.info(f"Train class distribution (encoded): {Counter(y_train_encoded)}")
        if X_test_orig.shape[0] > 0: logger.info(f"Test class distribution (encoded): {Counter(y_test_encoded)}")

        # Scaling: Fit on X_train_orig only, transform all
        scaler = StandardScaler() #MinMaxScaler()
        X_train_scaled_np = np.array([]).reshape(0, X_train_orig.shape[1])
        if X_train_orig.shape[0] > 0:
            X_train_scaled_np = scaler.fit_transform(X_train_orig)
        
        X_test_scaled_np = scaler.transform(X_test_orig) if X_test_orig.shape[0] > 0 else X_test_orig
        X_holdout_scaled_np = scaler.transform(X_holdout_orig) if X_holdout_orig.shape[0] > 0 else X_holdout_orig
        
    # Universal Finalization Step for ALL Datasets ---  
    logger.info("Finalizing data preparation...")    
    y_train_final_np, y_test_final_np, y_holdout_final_np = None, None, None  
    target_tensor_dtype = None
    if not is_binary and num_classes_found > 1: # Multiclass
        target_tensor_dtype = torch.long
        y_train_final_np = y_train_encoded.astype(np.int64)
        y_test_final_np = y_test_encoded.astype(np.int64) if X_test_orig.shape[0] > 0 else np.array([], dtype=np.int64)
        y_holdout_final_np = y_holdout_encoded.astype(np.int64) if X_holdout_orig.shape[0] > 0 else np.array([], dtype=np.int64)
    else: # Binary mode
        target_tensor_dtype = torch.float32
        y_train_final_np = y_train_encoded.astype(np.float32) # y_encoded is already 0 or 1 from LabelEncoder if binarization worked
        y_test_final_np = y_test_encoded.astype(np.float32) if X_test_orig.shape[0] > 0 else np.array([], dtype=np.float32)
        y_holdout_final_np = y_holdout_encoded.astype(np.float32) if X_holdout_orig.shape[0] > 0 else np.array([], dtype=np.float32)


    data_dict = {
        # PyTorch Tensors for training/evaluation (scaled X, final type y)
        'X_train': torch.tensor(X_train_scaled_np, dtype=torch.float32) if X_train_scaled_np.shape[0] > 0 else torch.empty(0, X_raw.shape[1], dtype=torch.float32),
        'y_train': torch.tensor(y_train_final_np, dtype=target_tensor_dtype) if y_train_final_np.shape[0] > 0 else torch.empty(0, dtype=target_tensor_dtype),
        
        'X_test': torch.tensor(X_test_scaled_np, dtype=torch.float32) if X_test_scaled_np.shape[0] > 0 else torch.empty(0, X_raw.shape[1], dtype=torch.float32),
        'y_test': torch.tensor(y_test_final_np, dtype=target_tensor_dtype) if y_test_final_np.shape[0] > 0 else torch.empty(0, dtype=target_tensor_dtype),
        
        'X_holdout': torch.tensor(X_holdout_scaled_np, dtype=torch.float32) if X_holdout_scaled_np.shape[0] > 0 else torch.empty(0, X_raw.shape[1], dtype=torch.float32),
        'y_holdout': torch.tensor(y_holdout_final_np, dtype=target_tensor_dtype) if y_holdout_final_np.shape[0] > 0 else torch.empty(0, dtype=target_tensor_dtype),

        # NumPy arrays - Original (unscaled) features
        'X_train_np_orig': X_train_orig,
        'X_test_np_orig': X_test_orig,
        'X_holdout_np_orig': X_holdout_orig, 

        # NumPy arrays - Scaled features
        'X_train_np_scaled': X_train_scaled_np,
        'X_test_np_scaled': X_test_scaled_np,
        'X_holdout_np_scaled': X_holdout_scaled_np,

        # NumPy arrays - Encoded targets (final dtype for model)
        'y_train_np': y_train_final_np, # Already has the correct dtype for loss
        'y_test_np': y_test_final_np,
        'y_holdout_np': y_holdout_final_np, # This can be used for saving y_holdout.npy

        # Metadata and objects
        'feature_names': final_feature_names, # feature_names,
        'scaler': scaler,
        'label_encoder': le,
        'is_binary_mode': is_binary,
        'num_classes': len(np.unique(y_train_final_np)), #num_classes_found,
        'has_channel_dim_model_input': has_channel_dim_output, # if models should unsqueeze channel dim
        'feature_dim': X_train_scaled_np.shape[1], #X_raw.shape[1], # Original feature dimension before scaling
        'metrics_group_names': final_group_metrics,
        'num_zones': num_features_per_group
    }
    logger.info(f"Data preparation completed. Target tensor dtype for model: {target_tensor_dtype}")
    return data_dict


# --- 7. MODEL COMPARISON ---
def compare_models_visualizations(all_model_metrics_eval, report_dir, subdirs):
    """Compare multiple models based on their evaluation metrics from evaluate_model."""
    logger.info("Comparing models...")
    # all_model_metrics_eval should be a dict: {'model_name': eval_metrics_dict, ...}

    model_names = list(all_model_metrics_eval.keys())
    if not model_names:
        logger.info("No model metrics found to compare.")
        return pd.DataFrame()

    metric_keys = ['accuracy', 'f1', 'precision', 'recall', 'auc']
    summary_data = []

    fpr_list, tpr_list, auc_list_roc, names_list_roc = [], [], [], []

    for name in model_names:
        metrics = all_model_metrics_eval[name]
        row = {'Model': name}
        for key in metric_keys:
            row[key] = metrics.get(key, 0)
        summary_data.append(row)

        if metrics.get('fpr') and metrics.get('tpr') and metrics.get('auc') is not None:
            fpr_list.append(metrics['fpr'])
            tpr_list.append(metrics['tpr'])
            auc_list_roc.append(metrics['auc'])
            names_list_roc.append(name)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(subdirs['metrics'] / "model_comparison_summary.csv", index=False)
    logger.info(f"Model comparison summary saved to {subdirs['metrics'] / 'model_comparison_summary.csv'}")


    # Bar chart of key metrics
    if not summary_df.empty:
        summary_df.set_index('Model').plot(kind='bar', figsize=(12, 7), colormap='viridis')
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='lower right')
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        save_figure(plt.gcf(), "model_comparison_all_metrics_bar", subdirs['figures'])


    # Combined ROC Curve
    if fpr_list and tpr_list:
        fig_roc_comp = plot_roc_curve(fpr_list, tpr_list, auc_list_roc, names_list_roc)
        save_figure(fig_roc_comp, "model_comparison_roc_curves", subdirs['figures'])


    # Confusion Matrices Side-by-Side
    num_models_cm = len([name for name in model_names if 'confusion_matrix' in all_model_metrics_eval[name]])
    if num_models_cm > 0:
        # Determine grid size (e.g., try to make it squarish)
        cols_cm = int(np.ceil(np.sqrt(num_models_cm)))
        rows_cm = int(np.ceil(num_models_cm / cols_cm))
        fig_cm_comp, axes_cm = plt.subplots(rows_cm, cols_cm, figsize=(cols_cm * 5, rows_cm * 4.5), squeeze=False)
        axes_cm_flat = axes_cm.flatten()
        plot_idx = 0
        for i, name in enumerate(model_names):
            if 'confusion_matrix' in all_model_metrics_eval[name] and all_model_metrics_eval[name]['confusion_matrix']:
                cm = np.array(all_model_metrics_eval[name]['confusion_matrix'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm_flat[plot_idx], cbar=False)
                axes_cm_flat[plot_idx].set_title(f'{name}')
                axes_cm_flat[plot_idx].set_xlabel('Predicted')
                axes_cm_flat[plot_idx].set_ylabel('True')
                plot_idx +=1

        # Hide any unused subplots
        for j in range(plot_idx, len(axes_cm_flat)):
            fig_cm_comp.delaxes(axes_cm_flat[j])

        plt.tight_layout()
        save_figure(fig_cm_comp, "model_comparison_confusion_matrices", subdirs['figures'])

    return summary_df



# --- 8A. STATISTICAL ANALYSIS FUNCTIONS ---
def calculate_f1_ci(y_true, y_pred, n_bootstraps=1000, ci_level=0.95):
    """Calculates the F1 score and its confidence interval using bootstrapping."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    f1_scores = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue # Skip bootstrap sample if it only has one class
            
        f1 = f1_score(y_true[indices], y_pred[indices], average='binary', zero_division=0)
        f1_scores.append(f1)
        
    if not f1_scores:
        return (np.nan, np.nan, np.nan)

    original_f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    lower_percentile = (1.0 - ci_level) / 2.0 * 100
    upper_percentile = (1.0 + ci_level) / 2.0 * 100
    
    lower_bound = np.percentile(f1_scores, lower_percentile)
    upper_bound = np.percentile(f1_scores, upper_percentile)
    
    return original_f1, lower_bound, upper_bound

def perform_mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Performs McNemar's test to compare the predictions of two models.

    Args:
        y_true: True labels.
        y_pred1: Predictions from model 1.
        y_pred2: Predictions from model 2.

    Returns:
        A dictionary containing the contingency table and the p-value.
    """
    y_true = np.asarray(y_true)
    y_pred1 = np.asarray(y_pred1)
    y_pred2 = np.asarray(y_pred2)

    # Contingency table cells
    model1_correct_model2_correct = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
    model1_correct_model2_incorrect = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    model1_incorrect_model2_correct = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    model1_incorrect_model2_incorrect = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))

    contingency_table = [[model1_correct_model2_correct, model1_correct_model2_incorrect],
                         [model1_incorrect_model2_correct, model1_incorrect_model2_incorrect]]

    # The cells for McNemar's test are b and c
    b = model1_correct_model2_incorrect
    c = model1_incorrect_model2_correct
    
    # Check for zero division if models are identical
    if b + c == 0:
        p_value = 1.0
    else:
        # Calculate McNemar's statistic and p-value
        # Using continuity correction
        statistic = abs(b - c)**2 / (b + c)
        p_value = 1 - chi2.cdf(statistic, df=1)

    return {"contingency_table": contingency_table, "p_value": p_value}

# --- 8. POST-TRAINING ANALYSIS ON HOLDOUT ---
def post_training_analysis(model_path, scaler_path, label_encoder_path, holdout_data_paths,
                           model_class_creator, model_params_creator, criterion, device,
                           report_dir, subdirs, # subdirs here should be for post_analysis outputs
                           feature_names, 
                           has_channel_dim_model_input, # Boolean for shaping X_holdout_np_scaled
                           is_transformer_model_heuristic_for_shap, 
                           X_train_np_scaled_for_shap_background, # This is the background data
                           is_binary, num_classes,
                           group_metrics_list, num_zones_per_metric,
                           shap_num_samples_from_args, # Value from parsed_args.shap_num_samples
                           model_expects_channel_dim_actual_func # The actual helper function
                          ):
    logger.info(f"--- Starting Post-Training Analysis on Holdout Set for model: {Path(model_path).stem} ---")
    post_analysis_dir = subdirs['post_analysis'] # subdirs from main usually contains this key
    post_analysis_figures_dir = post_analysis_dir / 'figures'
    post_analysis_metrics_dir = post_analysis_dir / 'metrics'
    post_analysis_figures_dir.mkdir(parents=True, exist_ok=True)
    post_analysis_metrics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    model = model_class_creator(**model_params_creator).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        return

    # 2. Load scaler and label encoder
    try:
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler from {scaler_path}")
        le = joblib.load(label_encoder_path)
        logger.info(f"Loaded label encoder from {label_encoder_path}")
        loaded_class_labels = list(le.classes_)
    except Exception as e:
        logger.error(f"Failed to load scaler/encoder: {e}")
        return

    # 3. Load holdout data
    try:
        X_holdout_np_orig = np.load(holdout_data_paths['X_orig'])
        y_holdout_np = np.load(holdout_data_paths['y'])
        logger.info(f"Loaded holdout data: X_holdout shape {X_holdout_np_orig.shape}, y_holdout shape {y_holdout_np.shape}")
    except Exception as e:
        logger.error(f"Failed to load holdout data: {e}")
        return

    if X_holdout_np_orig.shape[0] == 0:
        logger.info("Holdout set is empty. Skipping post-training analysis.")
        return

    # 4. Preprocess holdout data
    X_holdout_scaled_np = scaler.transform(X_holdout_np_orig)
    X_holdout_tensor = torch.tensor(X_holdout_scaled_np, dtype=torch.float32)
    
    target_dtype_holdout = torch.long if not is_binary and num_classes > 1 else torch.float32
    y_holdout_tensor_prepared = torch.tensor(y_holdout_np, dtype=target_dtype_holdout).to(device)

    if is_binary:
        if len(y_holdout_tensor_prepared.shape) == 1 and y_holdout_tensor_prepared.numel() > 0:
            y_holdout_tensor_prepared = y_holdout_tensor_prepared.unsqueeze(1)
    
    if has_channel_dim_model_input:
        if len(X_holdout_tensor.shape) == 2 and X_holdout_tensor.numel() > 0:
             X_holdout_tensor = X_holdout_tensor.unsqueeze(1)
    
    holdout_dataset = TensorDataset(X_holdout_tensor.to(device), y_holdout_tensor_prepared)
    holdout_loader = DataLoader(holdout_dataset, batch_size=16, shuffle=False) # Consider making batch_size an arg

    # 5. Evaluate model on holdout set
    logger.info("Evaluating model on holdout set...")
    model_name_holdout_eval = f"{Path(model_path).stem}_holdout_eval"
    
    # Define subdirectories for evaluate_model specifically for post_analysis
    eval_subdirs_post = {
        'figures': post_analysis_figures_dir, 
        'metrics': post_analysis_metrics_dir,
        'reports': post_analysis_metrics_dir # If evaluate_model saves text reports here
    }

    holdout_eval_metrics = evaluate_model(
        model=model, 
        test_loader=holdout_loader, 
        criterion=criterion, 
        device=device,
        model_name=model_name_holdout_eval,
        report_dir=post_analysis_dir, # Base for this specific evaluation
        subdirs=eval_subdirs_post,    # Specific subdirs within post_analysis
        is_binary=is_binary, 
        num_classes=num_classes,
        class_labels_for_plots=loaded_class_labels
    )

    if not holdout_eval_metrics or 'y_true' not in holdout_eval_metrics or 'y_pred' not in holdout_eval_metrics:
        logger.error(f"Evaluation on holdout set failed or did not return expected keys for {model_name_holdout_eval}. Skipping further analysis.")
        return

    # The evaluate_model function should ideally save its own full report.
    # If you need a separate JSON that evaluate_model doesn't provide:
    y_true_holdout = np.array(holdout_eval_metrics['y_true'])
    y_pred_holdout = np.array(holdout_eval_metrics['y_pred'])
    y_prob_holdout = np.array(holdout_eval_metrics['y_prob'] if 'y_prob' in holdout_eval_metrics else None)
    
    
    # if y_true_holdout.size > 0:
    #     np.save(post_analysis_metrics_dir / f"{model_name_holdout_eval}_y_true.npy", y_true_holdout)
    #     logger.info(f"Saved y_true array to {post_analysis_metrics_dir}")
    # if y_pred_holdout.size > 0:
    #     np.save(post_analysis_metrics_dir / f"{model_name_holdout_eval}_y_pred.npy", y_pred_holdout)
    #     logger.info(f"Saved y_pred array to {post_analysis_metrics_dir}")
    # if y_prob_holdout.size > 0:
    #     np.save(post_analysis_metrics_dir / f"{model_name_holdout_eval}_y_prob.npy", y_prob_holdout)
    #     logger.info(f"Saved y_prob array to {post_analysis_metrics_dir}")

    if len(y_true_holdout) > 0 and len(y_pred_holdout) > 0:
        logger.info(f"Holdout Evaluation Metrics for {Path(model_path).stem}: "
                    f"Acc: {holdout_eval_metrics.get('accuracy', -1):.4f}, "
                    f"F1: {holdout_eval_metrics.get('f1', -1):.4f}, "
                    f"AUC: {holdout_eval_metrics.get('roc_auc', -1):.4f}") # Use roc_auc from metrics
        try:
            holdout_class_report_dict = classification_report(
                y_true_holdout, y_pred_holdout, 
                target_names=[str(cls) for cls in loaded_class_labels], # Use loaded labels
                output_dict=True, zero_division=0
            )
            
            # ---- Calculate and save F1 Confidence Interval ----
            f1_score_val, f1_lower, f1_upper = calculate_f1_ci(y_true_holdout, y_pred_holdout)
            
            # Add the prediction lists to the report dictionary
            holdout_class_report_dict['y_true'] = y_true_holdout.tolist()
            holdout_class_report_dict['y_pred'] = y_pred_holdout.tolist()
            holdout_class_report_dict['y_prob'] = y_prob_holdout.tolist()
            # Add F1 CI to the dictionary
            holdout_class_report_dict['f1_score_ci'] = {
                                            'f1_score': f1_score_val,
                                            'lower_bound_95_ci': f1_lower,
                                            'upper_bound_95_ci': f1_upper
                                        }
            
            report_path = post_analysis_metrics_dir / f"{model_name_holdout_eval}_classification_report.json"
            with open(report_path, 'w') as f:
                json.dump(holdout_class_report_dict, f, indent=4)
            # logger.info(f"Saved detailed holdout classification report to {report_path}")
            logger.info(f"Calculated F1 Score CI: {f1_score_val:.4f} (95% CI: [{f1_lower:.4f}, {f1_upper:.4f}])")
            logger.info(f"Updated classification report with F1 CI at {report_path}")
      
        except Exception as e_rep:
            logger.warning(f"Could not generate/save detailed classification report for holdout: {e_rep}")
    else:
        logger.warning("No y_true or y_pred found in holdout_eval_metrics for detailed report.")


    # 6. Run SHAP analysis on the entire holdout set
    logger.info("Running SHAP analysis on the full holdout set...")
    shap_subdirs_post_main = {
        'figures': post_analysis_figures_dir, # SHAP plots go into post_analysis/figures/SHAP/...
        'data': subdirs.get('data') # Pass data subdir if SHAP needs to save anything there
    }
    # X_holdout_scaled_np is created a few steps above this in your function
    n_explain = 10 #200
    if X_holdout_scaled_np.shape[0] > n_explain:
        logger.info(f"For post-analysis, explaining a stratified sample of {n_explain} holdout instances.")
        
        # Use y_holdout_np for stratification
        _, X_explain_subset_np, _, _ = train_test_split(
            X_holdout_scaled_np, 
            y_holdout_np, 
            test_size=n_explain, 
            stratify=y_holdout_np, 
            random_state=42 # Using a fixed seed for this analysis
        )
    else:
        X_explain_subset_np = X_holdout_scaled_np
    # ------
    
    run_shap_analysis_dl(
        model=model,
        X_data_np=X_explain_subset_np, #X_holdout_scaled_np, # Pass the (potentially reshaped) holdout data
        X_train_data_np_for_background=X_train_np_scaled_for_shap_background,
        model_name_prefix=f"{model_path.stem}_holdout_full", # Clearer prefix
        subdirs=shap_subdirs_post_main, # These are subdirs for post_analysis
        feature_names=feature_names,
        is_binary_mode_flag=is_binary,
        num_classes_for_multiclass=num_classes,
        is_transformer_model_heuristic_for_shap=is_transformer_model_heuristic_for_shap,
        shap_num_samples_cli=shap_num_samples_from_args, # Use value from args
        group_metrics_list=group_metrics_list,
        num_zones_per_metric=num_zones_per_metric,
        model_expects_channel_dim_func=model_expects_channel_dim_actual_func, # Pass the function
        device=device,
        label_encoder_for_class_names=le if not is_binary else None,
        # label_encoder_for_class_names= num_classes,
        class_indices_to_explain=None, # Explain all classes or positive class
        explainer_type='kernel' #'gradient',# 'deep', or 'kernel'.
        
        
        # model=model,
        # X_data_np=X_holdout_scaled_np, # Explain the full holdout set
        # feature_names=feature_names,
        # model_name_prefix=f"{Path(model_path).stem}_holdout_full", # Distinct prefix
        # report_dir=report_dir, # Main report_dir
        # subdirs=shap_subdirs_post_main, 
        # device=device,
        # is_binary_mode_flag=is_binary,
        # num_actual_classes=num_classes,
        # class_indices_to_explain=None, # Explain all/default
        # is_transformer_model_heuristic=is_transformer_model_heuristic_for_shap, # Pass the flag
        # X_train_data_np_for_background=X_train_np_scaled_for_shap_background, # Pass training background
        # group_metrics_list=group_metrics_list,
        # num_zones_per_metric=num_zones_per_metric
    )
    
    # 7. Error analysis on misclassified instances from holdout
    logger.info("--- Starting Error Analysis on Misclassified Holdout Instances ---")
    misclassified_indices = np.where(y_true_holdout != y_pred_holdout)[0]

    if len(misclassified_indices) == 0:
        logger.info("No misclassified instances found in the holdout set. Skipping error analysis.")
    else:
        logger.info(f"Found {len(misclassified_indices)} misclassified instances in holdout.")
        
        num_misclassified_to_explain = min(len(misclassified_indices), 5) # Explain up to 5
        np.random.shuffle(misclassified_indices) 
        sample_misclassified_indices = misclassified_indices[:num_misclassified_to_explain]
        
        X_misclassified_sample_np = X_holdout_scaled_np[sample_misclassified_indices, :]

        if X_misclassified_sample_np.shape[0] > 0:
            error_analysis_shap_figures_dir = post_analysis_figures_dir / "SHAP_Error_Analysis"
            error_analysis_shap_figures_dir.mkdir(parents=True, exist_ok=True)
            
            shap_subdirs_post_error = {
                'figures': error_analysis_shap_figures_dir, # Specific figures subdir for error SHAP
                'data': subdirs.get('data')
            }

            logger.info(f"Running SHAP for {X_misclassified_sample_np.shape[0]} misclassified samples.")
            run_shap_analysis_dl(
                model=model,
                X_data_np=X_misclassified_sample_np,
                X_train_data_np_for_background=X_train_np_scaled_for_shap_background,
                feature_names=feature_names,
                model_name_prefix=f"{Path(model_path).stem}_holdout_misclassified",
                subdirs=shap_subdirs_post_error, 
                device=device,
                is_binary_mode_flag=is_binary,
                num_classes_for_multiclass=num_classes,
                is_transformer_model_heuristic_for_shap=is_transformer_model_heuristic_for_shap, shap_num_samples_cli=shap_num_samples_from_args,
                # For misclassified, often waterfall plots are most useful.
                # The run_shap_analysis_dl will generate its standard set of plots.
                group_metrics_list=group_metrics_list,
                num_zones_per_metric=num_zones_per_metric,
                model_expects_channel_dim_func=model_expects_channel_dim_actual_func,
                label_encoder_for_class_names=le if not is_binary else None,
                class_indices_to_explain=None,
                explainer_type='kernel' #'gradient', #'deep', or 'kernel'.
                )
            logger.info(f"SHAP plots for error analysis saved in {error_analysis_shap_figures_dir}")

            for i_idx, original_idx in enumerate(sample_misclassified_indices):
                logger.info(f"Misclassified instance (original holdout index {original_idx}, explained sample index {i_idx}): "
                            f"True={y_true_holdout[original_idx]}, Pred={y_pred_holdout[original_idx]}")
        else:
            logger.info("No misclassified samples selected for SHAP error analysis.")

    logger.info(f"--- Post-Training Analysis on Holdout Set Complete for {Path(model_path).stem} ---")
    
    # Calculate F1 CI for the holdout set
    f1_score_val, f1_lower, f1_upper = calculate_f1_ci(y_true_holdout, y_pred_holdout)

    return {
        'holdout_acc': holdout_eval_metrics.get('accuracy', -1),
        'holdout_f1': f1_score_val,
        'holdout_f1_ci': f"[{f1_lower:.4f}, {f1_upper:.4f}]",
        'holdout_kappa': holdout_eval_metrics.get('kappa', -1), # <-- ADD THIS
        'holdout_y_pred': y_pred_holdout,
        'holdout_y_true': y_true_holdout
    }



# --- MAIN FUNCTION ---
def main(args):
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    logger.info(f"Using device: {device}")

    report_dir, subdirs = create_report_directory(base_dir=args.report_base_dir, experiment_name=args.experiment_name)
    
    config_to_save = vars(parsed_args).copy()
    config_to_save['tune_hyperparameters'] = parsed_args.tune_hyperparameters
    save_experiment_config(config_to_save, report_dir)

    # --- 1. Data Preparation ---
    data_dict = prepare_data_for_dl(
        data_path=args.data_path,
        is_binary=args.binary_classification,
        dataset_name=args.dataset,
        stratify_split=args.stratify_split,
        test_size=args.test_split_ratio,
        holdout_size=args.holdout_split_ratio,
        random_state=args.seed,
        selected_features=args.select_features,
        test_data_path=args.test_data_path,
        resampling_method=args.resampling_method 
    )
    # ... (Saving holdout data and scaler/encoder)
    if data_dict.get('X_holdout_np_orig') is not None and data_dict['X_holdout_np_orig'].shape[0] > 0:
        np.save(subdirs['data'] / "X_holdout_orig.npy", data_dict['X_holdout_np_orig'])
        np.save(subdirs['data'] / "y_holdout.npy", data_dict['y_holdout_np'])
        logger.info(f"Saved holdout data to {subdirs['data']}")
    joblib.dump(data_dict['scaler'], subdirs['data'] / "scaler.joblib")
    joblib.dump(data_dict['label_encoder'], subdirs['data'] / "label_encoder.joblib")
    logger.info(f"Saved scaler and label encoder to {subdirs['data']}")

    X_train_full = data_dict['X_train']
    y_train_full = data_dict['y_train']
    X_test_eval = data_dict['X_test']
    y_test_eval = data_dict['y_test']
    feature_dim = data_dict['feature_dim']
    has_channel_dim_model_input = data_dict['has_channel_dim_model_input']
    
    num_classes = data_dict['num_classes']
    logger.info(f"Data preparation complete. Number of classes: {num_classes}.")
    if args.binary_classification and num_classes > 2:
        logger.warning(f"Binary mode selected, but {num_classes} classes found. Target was binarized.")
    elif not args.binary_classification and num_classes <= 1:
        raise ValueError(f"Multiclass mode, but {num_classes} classes found. Need > 1.")
    criterion = nn.BCEWithLogitsLoss() if args.binary_classification else nn.CrossEntropyLoss()
    
    def model_expects_channel_dim(mod):
        if isinstance(mod, (CNNModel, CNNTransformerModel)):
            return True
        if isinstance(mod, TransformerModel):
            # Check a config if your TransformerModel has one, otherwise default
            return getattr(mod, 'config', {}).get('expect_channel_dim', False)
        return False

    # Loss function
    if args.dataset == 'scania_aps':
        # Calculate pos_weight for the highly imbalanced dataset
        # pos_weight = count(negative) / count(positive)
        y_train_series = pd.Series(data_dict['y_train_np'])
        counts = y_train_series.value_counts()
        pos_weight_value = counts[0] / counts[1]
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(device))
        logger.info(f"Using BCEWithLogitsLoss with dynamically calculated pos_weight: {pos_weight_value:.2f}.")
    # elif args.binary_classification: 
    elif not args.binary_classification and num_classes > 1: #multiclass
        criterion = nn.CrossEntropyLoss()
        logger.info(f"Using CrossEntropyLoss for multiclass ({num_classes} classes).")
    elif (args.binary_classification and num_classes == 2):  
        # --- START OF MODIFICATION ---
        # Check if a resampling method was specified via command-line arguments
        if args.resampling_method:
            # If resampling is used, pos_weight can be 1.0 (no weighting needed)
            pos_weight_tensor = torch.tensor([1.0], dtype=torch.float32)
            logger.info(f"Using BCEWithLogitsLoss with pos_weight 1.0 because resampling is enabled.")
        else:
            # If no resampling, use the dynamic pos_weight
            y_train_series = pd.Series(data_dict['y_train_np'])
            counts = y_train_series.value_counts()
            pos_weight_value = counts[0] / counts[1] if 1 in counts and counts[1] > 0 else 1.0
            pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32)
            logger.info(f"Using BCEWithLogitsLoss with dynamically calculated pos_weight: {pos_weight_value:.2f}.")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(device))
        # --- END OF MODIFICATION ---
               
    #     pos_weight_value_float = 1.5 
    #     pos_weight_tensor = torch.tensor([pos_weight_value_float], dtype=torch.float32)
    #     criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor.to(device))
    #     logger.info(f"Using BCEWithLogitsLoss with HARDCODED pos_weight: {pos_weight_tensor.item():.2f}.")
    # elif args.binary_classification and num_classes == 1 :
    #     logger.error("Binary mode, but only 1 class found after processing.")
    #     raise ValueError("Binary mode, 1 class found. Check data.")
    else: 
        logger.error(f"Inconsistent args: binary={args.binary_classification}, classes={num_classes}.")
        raise ValueError("Cannot determine loss function.")
    # 
    # 
  
    # ==============================================================================
    # --- STAGE 1: MODEL TRAINING & VALIDATION ---
    # ==============================================================================
    experiment_summary_results = []
    all_trained_model_paths = {}
    all_final_model_eval_metrics = {} # For the simple comparison summary
    model_architectures_to_run = args.models_to_run 
    logger.info(f"Models to run: {model_architectures_to_run}")

    # Get scaled training data for SHAP background once
    X_train_np_scaled_for_shap_bg = data_dict.get('X_train_np_scaled', None)
    if X_train_np_scaled_for_shap_bg is None:
        logger.warning(f"Scaled training data ('X_train_np_scaled') not found in data_dict. "
                       "SHAP analysis will use explanation data as background (less ideal).")

    for model_type in model_architectures_to_run:
        logger.info(f"\n--- Processing Model Type: {model_type} ---")
        
        # Determine the heuristic flag for SHAP based on model_type
        is_transformer_heuristic_flag_for_shap = "transformer" in model_type.lower() 
        
        model_output_classes = 1 if args.binary_classification else num_classes
        base_constructor_params = {'feature_dim': feature_dim, 'num_classes': model_output_classes}
        current_model_specific_params = {'dropout_rate': args.dropout_rate}
        current_optimizer_params = {'lr': args.learning_rate, 'weight_decay': args.weight_decay}

        model_class_ref = None
        if model_type == 'CNN':
            model_class_ref = CNNModel
            current_model_specific_params.update({'cnn_units': args.cnn_units})
        elif model_type == 'Transformer':
            model_class_ref = TransformerModel
            current_model_specific_params.update({
                'transformer_dim': args.transformer_dim,
                'transformer_heads': args.transformer_heads,
                'transformer_layers': args.transformer_layers
            })
        elif model_type == 'CNNTransformer_sequential': 
            model_class_ref = CNNTransformerModel
            current_model_specific_params.update({
                'cnn_units': args.cnn_units_hybrid if hasattr(args, 'cnn_units_hybrid') else args.cnn_units,
                'transformer_dim': args.transformer_dim_hybrid if hasattr(args, 'transformer_dim_hybrid') else args.transformer_dim,
                'transformer_heads': args.transformer_heads_hybrid if hasattr(args, 'transformer_heads_hybrid') else args.transformer_heads,
                'transformer_layers': args.transformer_layers_hybrid if hasattr(args, 'transformer_layers_hybrid') else args.transformer_layers,
                'architecture_mode': 'sequential',
                'fc_dropout': args.fc_dropout if hasattr(args, 'fc_dropout') else args.dropout_rate,
                'cnn_dropout': args.cnn_dropout if hasattr(args, 'cnn_dropout') else args.dropout_rate,
                'transformer_dropout': args.transformer_dropout if hasattr(args, 'transformer_dropout') else args.dropout_rate,
            })
        elif model_type == 'CNNTransformer_parallel': 
            model_class_ref = CNNTransformerModel
            current_model_specific_params.update({
                'cnn_units': args.cnn_units_hybrid if hasattr(args, 'cnn_units_hybrid') else args.cnn_units,
                'transformer_dim': args.transformer_dim_hybrid if hasattr(args, 'transformer_dim_hybrid') else args.transformer_dim,
                'transformer_heads': args.transformer_heads_hybrid if hasattr(args, 'transformer_heads_hybrid') else args.transformer_heads,
                'transformer_layers': args.transformer_layers_hybrid if hasattr(args, 'transformer_layers_hybrid') else args.transformer_layers,
                'architecture_mode': 'parallel',
                'fc_dropout': args.fc_dropout if hasattr(args, 'fc_dropout') else args.dropout_rate,
                'cnn_dropout': args.cnn_dropout if hasattr(args, 'cnn_dropout') else args.dropout_rate,
                'transformer_dropout': args.transformer_dropout if hasattr(args, 'transformer_dropout') else args.dropout_rate,
            })
        else:
            logger.warning(f"Unknown model type: {model_type}. Skipping.")
            continue
        
        if args.tune_hyperparameters:
            logger.info(f"--- Hyperparameter Tuning for {model_type} ---")
            param_grid_for_tuning = {}
            if model_type == 'CNN':
                param_grid_for_tuning = {
                    # 'cnn_units': [32, 64, 128], 'dropout_rate': [0.2, 0.3, 0.5],
                    # 'learning_rate': [1e-5, 1e-4, 1e-3], 'weight_decay': [1e-5, 1e-4, 1e-3] # wider range
                    'cnn_units': [64, 128],             
                    'dropout_rate': [0.2, 0.3],         
                    'learning_rate': [1e-4, 1e-3],      
                    'weight_decay': [1e-5, 1e-4, 1e-3],
                    'pos_weight_value': [1.0, 59.0, 75.0, 100.0, 200.0]                   
                }
                #debugging with smaller grid
            # if model_type == 'CNN':
            #     param_grid_for_tuning = {
            #         # 'cnn_units': [32, 64, 128], 'dropout_rate': [0.2, 0.3, 0.5],
            #         # 'learning_rate': [1e-5, 1e-4, 1e-3], 'weight_decay': [1e-5, 1e-4, 1e-3] # wider range
            #         'cnn_units': [64, 128],             
            #         'dropout_rate': [0.2],         
            #         'learning_rate': [1e-4],      
            #         'weight_decay': [1e-5]                   
            #     }    
                
            elif model_type == 'Transformer':
                param_grid_for_tuning = {
                    # 'transformer_dim': [64, 128], 'transformer_layers': [1, 2, 3],
                    # 'transformer_heads': [2, 4, 8], 'dropout_rate': [0.1, 0.2, 0.3],
                    # 'learning_rate': [1e-5, 1e-4, 1e-3], 'weight_decay': [1e-5, 1e-4, 1e-3] # wider range
                    # 'transformer_dim': [64],            # Fix dimension to reduce combinations.
                    # 'transformer_layers': [1, 2],       # Test a shallow vs. slightly deeper model.
                    # 'transformer_heads': [2, 4],        # Focus on a reasonable number of heads.
                    # 'dropout_rate': [0.2, 0.3],
                    # 'learning_rate': [1e-4, 1e-3],
                    # 'weight_decay': [1e-5, 1e-4, 1e-3],
                    # 'pos_weight_value': [1.0, 59.0, 75.0, 100.0, 200.0],
                    #sample
                    'transformer_dim': [64, 128],
                    'transformer_layers': [1], # Keep it simple to avoid overfitting
                    'transformer_heads': [2], # Use a low number of heads
                    'dropout_rate': [0.2, 0.3],
                    'learning_rate': [1e-3, 1e-4], # Explore a lower learning rate
                    'weight_decay': [1e-4],
                    'pos_weight_value': [1.0, 59.0, 150.0, 300.0]
                }
            elif model_type == 'CNNTransformer_sequential' or model_type == 'CNNTransformer_parallel':
                param_grid_for_tuning = {
                    # 'cnn_units': [32, 64], 'transformer_dim': [64, 128],
                    # 'transformer_layers': [1], 'transformer_heads': [2],
                    # 'cnn_dropout': [0.2, 0.3, 0.5], 'transformer_dropout': [0.2, 0.3, 0.5],
                    # 'fc_dropout': [0.2, 0.3, 0.5],
                    # 'learning_rate': [1e-3], 'weight_decay': [1e-5, 1e-3]
                # sample 2
                    # 'cnn_units': [32, 64, 128],  # Smaller than before
                    # 'transformer_dim': [32, 64],  # Ensure divisible by heads
                    # 'transformer_layers': [1, 2],
                    # 'transformer_heads': [ 2],  # Fewer heads
                    # 'cnn_dropout': [0.2, 0.3], #, 0.5],  # Explore higher dropout
                    # 'transformer_dropout': [0.2, 0.3], #0.5],  # Explore higher dropout
                    # 'fc_dropout': [0.3, 0.2], #, 0.5],  # Explore higher dropout for the final layer
                    # 'learning_rate': [1e-3],
                    # 'weight_decay': [1e-4],  #[1e-3, 5e-3, 1e-3]  # Explore stronger L2 regularization
                    # 'pos_weight_value': [1.0] #, 59.0, 75.0, 100.0, 200.0]    
                    # sample
                        'cnn_units': [32, 64],
                        'transformer_dim': [32, 64],
                        'transformer_layers': [1],
                        'transformer_heads': [2],
                        'cnn_dropout': [0.2, 0.3],
                        'transformer_dropout': [0.2, 0.3],
                        'fc_dropout': [0.2, 0.3],
                        'learning_rate': [1e-3], # Keep this fixed
                        'weight_decay': [1e-4],
                        'pos_weight_value': [1.0, 59.0, 75.0] 
                }   
            
            # if args.binary_classification:
                # param_grid_for_tuning['pos_weight_value'] = [1.0, 59.0, 75.0, 100.0, 200.0]         
            
            # --- START of the parallel/sequential execution block ---
            if param_grid_for_tuning:
                param_combinations = list(ParameterGrid(param_grid_for_tuning))
                logger.info(f"Number of total parameter combinations: {len(param_combinations)}")
            
                num_gpus = torch.cuda.device_count()
                if num_gpus > 1:
                    logger.info(f"Found {num_gpus} GPUs. Distributing grid search...")
                    
                    chunks = [param_combinations[i::num_gpus] for i in range(num_gpus)]
                    worker_result_files = []
                    processes = []
                    
                    for i in range(num_gpus):
                        worker_file_path = subdirs['metrics'] / f"{model_type}_grid_search_worker_{i}.csv"
                        worker_result_files.append(worker_file_path)

                        # mp.set_start_method('spawn', force=True)
                        p = mp.Process(
                            target=grid_search_worker_wrapper,
                            args=(
                                i, chunks[i], model_type,
                                {**base_constructor_params, **current_model_specific_params},
                                data_dict['X_train_np_orig'], data_dict['y_train_np'],
                                current_optimizer_params, criterion, report_dir, subdirs,
                                feature_dim, has_channel_dim_model_input,
                                args.cv_splits_tuning, args.epochs_tuning, args.batch_size,
                                args.binary_classification
                            )
                        )
                        processes.append(p)
                        p.start()

                    for p in processes:
                        p.join()
                    
                    all_results_df = pd.concat([pd.read_csv(f) for f in worker_result_files])
                    
                    all_results_df = all_results_df.sort_values(by='avg_f1', ascending=False)
                    best_params_combo = all_results_df.iloc[0].to_dict()
                    best_tuning_score = best_params_combo.get('avg_f1', -1.0)
                    
                    all_results_df.to_csv(subdirs['metrics'] / f"{model_type}_grid_search_results.csv", index=False)
                    logger.info(f"Combined all grid search results and saved to {subdirs['metrics'] / f'{model_type}_grid_search_results.csv'}")

                    logger.info(f"Tuning for {model_type}: Best CV F1={best_tuning_score:.4f}")
                    
                    # --- ADD THIS LOGIC HERE TO HANDLE TYPE CASTING ---
                    final_tuned_params = {k: v for k, v in best_params_combo.items() if k not in ['learning_rate', 'weight_decay', 'avg_f1', 'std_f1']}
                    
                    # Manually cast model-specific parameters to the correct types
                    if 'cnn_units' in final_tuned_params:
                        final_tuned_params['cnn_units'] = int(final_tuned_params['cnn_units'])
                    if 'transformer_dim' in final_tuned_params:
                        final_tuned_params['transformer_dim'] = int(final_tuned_params['transformer_dim'])
                    if 'transformer_heads' in final_tuned_params:
                        final_tuned_params['transformer_heads'] = int(final_tuned_params['transformer_heads'])
                    if 'transformer_layers' in final_tuned_params:
                        final_tuned_params['transformer_layers'] = int(final_tuned_params['transformer_layers'])
                    # --- END OF ADDITION ---
                    
                    current_model_specific_params.update({k: v for k, v in best_params_combo.items() if k not in ['learning_rate', 'weight_decay', 'avg_f1', 'std_f1']})
                    current_optimizer_params.update({
                        'lr': best_params_combo.get('learning_rate', current_optimizer_params['lr']),
                        'weight_decay': best_params_combo.get('weight_decay', current_optimizer_params['weight_decay'])
                    })
                else:
                    # Fallback for single GPU
                    initial_params_for_gs = {**base_constructor_params, **current_model_specific_params}
                    tuned_model_hparams, tuned_optim_hparams, best_tuning_score = grid_search_tuning(
                        model_class=model_class_ref,
                        base_model_params=initial_params_for_gs,
                        param_grid_list=param_combinations,
                        X_train_gs=data_dict['X_train_np_orig'], y_train_gs=data_dict['y_train_np'],
                        optimizer_base_params=current_optimizer_params,
                        criterion_gs=criterion, device_gs=device,
                        gs_report_dir=report_dir, gs_subdirs=subdirs,
                        n_splits_gs=args.cv_splits_tuning, epochs_gs=args.epochs_tuning,
                        batch_size_gs=args.batch_size,
                        feature_dim_gs=feature_dim, has_channel_dim_gs=has_channel_dim_model_input,
                        is_binary_mode=args.binary_classification
                    )
                    logger.info(f"Tuning for {model_type}: Best CV F1={best_tuning_score:.4f}")
                    current_model_specific_params.update(tuned_model_hparams)
                    current_optimizer_params.update(tuned_optim_hparams)
            else:
                logger.info(f"No param grid for {model_type} tuning. Using defaults.")
            # --- END of the parallel/sequential execution block ---
            
            # if param_grid_for_tuning:
            #     initial_params_for_gs = {**base_constructor_params, **current_model_specific_params}
            #     tuned_model_hparams, tuned_optim_hparams, best_tuning_score = grid_search_tuning(
            #         model_class=model_class_ref, 
            #         base_model_params=initial_params_for_gs,
            #         param_grid=param_grid_for_tuning, 
            #         X_train_gs=X_train_full, 
            #         y_train_gs=y_train_full,
            #         optimizer_base_params=current_optimizer_params, 
            #         criterion_gs=criterion, device_gs=device,
            #         gs_report_dir=report_dir, gs_subdirs=subdirs,
            #         n_splits_gs=args.cv_splits_tuning, epochs_gs=args.epochs_tuning, 
            #         batch_size_gs=args.batch_size,
            #         feature_dim_gs= feature_dim,
            #         has_channel_dim_gs=has_channel_dim_model_input,
            #         is_binary_mode=args.binary_classification, 
            #         # num_model_output_classes=model_output_classes
            #     )
            #     logger.info(f"Tuning for {model_type}: Best CV F1={best_tuning_score:.4f}")
            #     current_model_specific_params.update(tuned_model_hparams)
            #     current_optimizer_params.update(tuned_optim_hparams)
            # else:
            #     logger.info(f"No param grid for {model_type} tuning. Using defaults.")
            
            logger.info(f"Params for final {model_type} training: Model={current_model_specific_params}, Optim={current_optimizer_params}")

        if model_type in ['CNNTransformer_sequential', 'CNNTransformer_parallel'] and 'architecture_mode' not in current_model_specific_params:
            logger.warning(f"Architecture_mode missing for {model_type}. Defaulting to {'sequential' if 'sequential' in model_type else 'parallel'}.")
            current_model_specific_params['architecture_mode'] = 'sequential' if 'sequential' in model_type else 'parallel'
            
        final_model_constructor_params = {**base_constructor_params, **current_model_specific_params}
        logger.info(f"For {model_type}, final constructor params: {final_model_constructor_params}")

        # --- Create a copy of the constructor params
        model_params_for_training = final_model_constructor_params.copy()
        # Remove the pos_weight_value, as it's only for the loss function
        if 'pos_weight_value' in model_params_for_training:
            model_params_for_training.pop('pos_weight_value')
        # ------
        
        
        

        final_model_path, _, avg_cv_metrics = train_model_with_cv(
            model_class=model_class_ref, model_params=model_params_for_training, #final_model_constructor_params,
            X_train_all=X_train_full, y_train_all=y_train_full, optimizer_params=current_optimizer_params,
            criterion=criterion, device=device, model_name_base=model_type, report_dir=report_dir, subdirs=subdirs,
            n_splits=args.cv_splits_training, epochs=args.epochs_training, batch_size=args.batch_size,
            has_channel_dim_input=has_channel_dim_model_input,
            is_binary=data_dict['is_binary_mode'], num_class=num_classes
        )
        
        all_trained_model_paths[model_type] = {
            'final_model': final_model_path, 
            'params': final_model_constructor_params.copy(),
            'avg_cv_metrics': avg_cv_metrics
        }
        
        
        # --- Evaluate on the test set for the simple comparison CSV ---
        # if data_dict['X_test'].numel() > 0:
        #     # ... (your existing logic to evaluate on the test set)
        #     eval_metrics = evaluate_model(...)
        #     all_final_model_eval_metrics[model_type] = eval_metrics
        if X_test_eval.numel() > 0 and y_test_eval.numel() > 0:
            logger.info(f"--- Evaluating final {model_type} on test set ---")
            # final_model_to_eval = model_class_ref(**final_model_constructor_params).to(device)
            final_model_to_eval = model_class_ref(**model_params_for_training).to(device)
            try:
                final_model_to_eval.load_state_dict(torch.load(final_model_path, map_location=device))
            except FileNotFoundError:
                logger.error(f"Model file {final_model_path} not found for eval.")
                continue 

            _X_test_eval_tensor = X_test_eval.to(device)
            if has_channel_dim_model_input and len(_X_test_eval_tensor.shape) == 2 and _X_test_eval_tensor.numel() > 0: 
                _X_test_eval_tensor = _X_test_eval_tensor.unsqueeze(1)
            
            _y_test_eval_tensor_prepared = y_test_eval.to(device=device)
            if args.binary_classification and len(_y_test_eval_tensor_prepared.shape) == 1 and _y_test_eval_tensor_prepared.numel() > 0:
                _y_test_eval_tensor_prepared = _y_test_eval_tensor_prepared.unsqueeze(1)
            
            test_dataset_eval = TensorDataset(_X_test_eval_tensor, _y_test_eval_tensor_prepared)
            test_loader_eval = DataLoader(test_dataset_eval, batch_size=args.batch_size, shuffle=False)
            
            eval_metrics = evaluate_model(
                model=final_model_to_eval, test_loader=test_loader_eval, criterion=criterion, device=device,
                model_name=f"{model_type}_final_eval_on_test", report_dir=report_dir, subdirs=subdirs,
                is_binary=args.binary_classification, num_classes=num_classes,
                class_labels_for_plots=list(data_dict['label_encoder'].classes_)
            )
            all_final_model_eval_metrics[model_type] = eval_metrics
        
            # --- Create the initial row for the summary CSV with validation results ---
            model_results_row = {
                'experiment_name': args.experiment_name,
                'model_name': model_type,
                'val_acc': avg_cv_metrics.get('accuracy', -1),
                'val_f1': avg_cv_metrics.get('f1_score_val', -1),
                'val_f1_95_ci': avg_cv_metrics.get('f1_95_ci_val', 'N/A'),
                    'holdout_acc': 'N/A',
                    'holdout_f1': 'N/A',
                    'holdout_f1_ci': 'N/A',
                'notes (params)': str(final_model_constructor_params)
            }
            experiment_summary_results.append(model_results_row)
            # SHap Analysis on Test Set
            logger.info(f"--- Running SHAP analysis for final {model_type} on test set ---")
            X_test_for_shap_np = data_dict.get('X_test_np_scaled')
            y_test_for_shap_np = data_dict.get('y_test_np') # <-- Get the corresponding labels
 
                
            if X_test_for_shap_np is not None and X_test_for_shap_np.shape[0] > 0:
                n_explain = 10 #200 # Choose a reasonable number, e.g., 200-500
                if X_test_for_shap_np.shape[0] > n_explain:
                    logger.info(f"Creating a stratified sample of {n_explain} instances instead of all {X_test_for_shap_np.shape[0]}.")

                    # Use train_test_split to create a stratified subsample.
                    # We only care about the 'test_size' part of the split, so we use placeholders '_' for the rest.
                    _, X_explain_subset_np, _, _ = train_test_split(
                        X_test_for_shap_np, 
                        y_test_for_shap_np, 
                        test_size=n_explain, 
                        stratify=y_test_for_shap_np, # This is the key to ensuring representation
                        random_state=args.seed
                    )
                else:
                    # If the test set is already small, use all of it
                    X_explain_subset_np = X_test_for_shap_np
                #
                
                run_shap_analysis_dl(
                    model=final_model_to_eval,
                    X_data_np=X_explain_subset_np, #X_test_for_shap_np
                    X_train_data_np_for_background=X_train_np_scaled_for_shap_bg, # Background from main scope
                    model_name_prefix=f"{model_type}_final_test",
                    subdirs=subdirs, # Main experiment subdirs
                    feature_names=data_dict['feature_names'],
                    is_binary_mode_flag=data_dict['is_binary_mode'],
                    num_classes_for_multiclass=num_classes, # num_classes from main scope
                    is_transformer_model_heuristic_for_shap=is_transformer_heuristic_flag_for_shap, # From main scope
                    shap_num_samples_cli=parsed_args.shap_num_samples, # From argparse
                    group_metrics_list=data_dict.get('metrics_group_names'),
                    num_zones_per_metric=data_dict.get('num_zones'),
                    model_expects_channel_dim_func=model_expects_channel_dim, # Pass the helper function
                    device=device,
                    label_encoder_for_class_names=data_dict['label_encoder'] if not data_dict['is_binary_mode'] else None,
                    # label_encoder if not data_dict['is_binary_mode'] else None,
                    class_indices_to_explain=None,
                    explainer_type='kernel' #'gradient', #'deep', or 'kernel'.
                    
                    # model=final_model_to_eval,
                    # X_data_np=X_test_for_shap_np,
                    # feature_names=data_dict['feature_names'],
                    # model_name_prefix=f"{model_type}_final_test", 
                    # report_dir=report_dir, subdirs=subdirs, device=device,
                    # is_binary_mode_flag=data_dict['is_binary_mode'], 
                    # num_actual_classes=num_classes,
                    # class_indices_to_explain=0 if not args.binary_classification and num_classes > 1 else None,
                    # group_metrics_list=data_dict.get('metrics_group_names'), 
                    # num_zones_per_metric=data_dict.get('num_zones'),
                    # is_transformer_model_heuristic=is_transformer_heuristic_flag_for_shap, # Pass the flag
                    # X_train_data_np_for_background=X_train_np_scaled_for_shap_bg # Pass training data for background
                )
            else:
                logger.info(f"Test set for SHAP (X_test_np_scaled) empty/not found for {model_type}. Skipping.")
        else:
                logger.info(f"No test set to evaluate or run SHAP for {model_type}.")

    if all_final_model_eval_metrics:
        logger.info("\n--- Comparing All Final Models (on Test Set) ---")
        compare_models_visualizations(all_final_model_eval_metrics, report_dir, subdirs)
    else:
        logger.info("No models evaluated on test set for comparison.")


    # ==============================================================================
    # --- STAGE 2: POST-ANALYSIS ON HOLDOUT SET ---
    # ==============================================================================
    all_holdout_results = {}
    if data_dict.get('X_holdout') is not None and data_dict['X_holdout'].numel() > 0 and all_trained_model_paths:
        logger.info("\n--- Starting Post-Training Holdout Analysis for All Trained Models ---")
        X_train_np_scaled_for_shap_bg = data_dict.get('X_train_np_scaled', None)

        # LOOP: Iterate through the results list to update each row
        for i, model_summary_row in enumerate(experiment_summary_results):
            model_key = model_summary_row['model_name']
            
            if model_key in all_trained_model_paths:
                model_info = all_trained_model_paths[model_key]
                path_to_final_model_for_post = model_info['final_model']
                params_for_post_model = model_info['params']
                # added to remove pos-weight from params if present
                model_params_for_post_analysis = params_for_post_model.copy()
                if 'pos_weight_value' in model_params_for_post_analysis:
                    model_params_for_post_analysis.pop('pos_weight_value')
                
                is_transformer_heuristic_for_post_analysis_call = "transformer" in model_key.lower()
                
                # Determine model class creator and other flags
                model_class_creator_post = None
                if model_key == 'CNN': model_class_creator_post = CNNModel
                elif model_key == 'Transformer': model_class_creator_post = TransformerModel
                # elif 'CNNTransformer' in model_key: model_class_creator_post = CNNTransformerModel
                elif model_key == 'CNNTransformer_sequential' or \
                 model_key == 'CNNTransformer_parallel':
                    model_class_creator_post = CNNTransformerModel
                    # Ensure 'architecture_mode' is in params_for_post_model
                    if 'architecture_mode' not in params_for_post_model:
                        logger.warning(f"Architecture_mode missing in stored params for {model_key}. Inferring from key.")
                        if 'sequential' in model_key:
                            params_for_post_model['architecture_mode'] = 'sequential'
                        elif 'parallel' in model_key:
                            params_for_post_model['architecture_mode'] = 'parallel'
                else:
                    logger.warning(f"Unknown model type {model_key} encountered during post-analysis loop. Skipping.")
                    continue
            
                logger.info(f"Model: {model_key}. Constructor Params for holdout analysis: {params_for_post_model}")
                
                if model_class_creator_post and Path(path_to_final_model_for_post).exists():
                    logger.info(f"\n--- Post-Processing for Model: {model_key} on Holdout Set ---")
                    
                    # Call post-analysis and get the results dictionary
                    holdout_results = post_training_analysis(
                    model_path=path_to_final_model_for_post,
                    scaler_path=subdirs['data'] / "scaler.joblib",
                    label_encoder_path=subdirs['data'] / "label_encoder.joblib",
                    holdout_data_paths={'X_orig': subdirs['data'] / "X_holdout_orig.npy",
                                        'y': subdirs['data'] / "y_holdout.npy"},
                    model_class_creator=model_class_creator_post,
                    model_params_creator=model_params_for_post_analysis, #params_for_post_model, 
                    criterion=criterion, 
                    device=device,
                    report_dir=report_dir, 
                    subdirs=subdirs,
                    feature_names=data_dict['feature_names'],
                    has_channel_dim_model_input=data_dict['has_channel_dim_model_input'],
                    is_transformer_model_heuristic_for_shap=is_transformer_heuristic_for_post_analysis_call,
                    X_train_np_scaled_for_shap_background=X_train_np_scaled_for_shap_bg,
                    is_binary=data_dict['is_binary_mode'], 
                    num_classes=num_classes, # num_classes from main scope
                    group_metrics_list=data_dict.get('metrics_group_names'),
                    num_zones_per_metric=data_dict.get('num_zones'),
                    shap_num_samples_from_args=parsed_args.shap_num_samples, # Pass arg value
                    model_expects_channel_dim_actual_func=model_expects_channel_dim # Pass the helper function
                )
                    
                    # Update the summary row with holdout results
                    if holdout_results:
                        update_dict = {
                            'holdout_acc': holdout_results['holdout_acc'],
                            'holdout_f1': holdout_results['holdout_f1'],
                            'holdout_f1_ci': holdout_results['holdout_f1_ci'],
                            'holdout_kappa': holdout_results.get('holdout_kappa', -1)
                        }
                        experiment_summary_results[i].update(update_dict)
                        all_holdout_results[model_key] = holdout_results
    elif not all_trained_model_paths:
         logger.info("No trained models found to perform post-training holdout analysis.")
    else: 
        logger.info("No holdout data available for post-training analysis.")

    # ==============================================================================
    # --- STAGE 3: FINAL COMPARISONS & REPORTING ---
    # ==============================================================================
    
    # --- Perform McNemar's Test on Holdout Predictions ---
    if len(all_holdout_results) >= 2:
        logger.info("\n--- Comparing Models with McNemar's Test on Holdout Set ---")
        best_model_name = max(all_holdout_results, key=lambda m: all_holdout_results[m]['holdout_f1'])
        y_true_holdout = np.array(all_holdout_results[best_model_name]['holdout_y_true'])

        for i, row in enumerate(experiment_summary_results):
            current_model_name = row['model_name']
            p_value_str = 'N/A'
            if current_model_name != best_model_name:
                y_pred1 = np.array(all_holdout_results[best_model_name]['holdout_y_pred'])
                y_pred2 = np.array(all_holdout_results[current_model_name]['holdout_y_pred'])
                mcnemar_res = perform_mcnemar_test(y_true_holdout, y_pred1, y_pred2)
                p_value_str = f"{mcnemar_res['p_value']:.4f}"
            elif current_model_name == best_model_name:
                 p_value_str = 'BEST_MODEL'
            experiment_summary_results[i]['mcnemar_p_vs_best_holdout'] = p_value_str
    
    # --- Create and Save the Final Summary CSV ---
    if experiment_summary_results:
        final_summary_df = pd.DataFrame(experiment_summary_results)
        summary_csv_path = subdirs['post_analysis'] / 'metrics' / f"{args.experiment_name}_final_summary.csv"
        
        # Reorder columns
        desired_order = [
            'experiment_name', 'model_name', 'val_acc', 'holdout_acc', 'val_f1', 'val_f1_95_ci',  'holdout_f1', 'holdout_f1_ci',  'holdout_kappa', 
             'mcnemar_p_vs_best_holdout', 'notes (params)'
        ]
        final_order = [col for col in desired_order if col in final_summary_df.columns]
        final_summary_df = final_summary_df[final_order]

        final_summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Saved final experiment summary to: {summary_csv_path}")

    # --- Save the simple test-set comparison CSV ---
    if all_final_model_eval_metrics:
        logger.info("\n--- Comparing Final Models on Initial Test Split ---")
        compare_models_visualizations(all_final_model_eval_metrics, report_dir, subdirs)

    logger.info(f"--- Experiment Complete. Reports saved in {report_dir} ---")





if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass # This handles cases where it might already be set
    
    parser = argparse.ArgumentParser(description="Deep Learning Classification Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the TRAINING data file.")
    parser.add_argument("--dataset", type=str, default="mpod", 
                        choices=["mpod", "breast_cancer", "scania_aps"], help="Identifier for the dataset in use, to apply correct feature handling.")
    
    parser.add_argument("--test_data_path", type=str, help="Path to the TEST data file (required if train/test data are already separated, e.g., for scania_aps).")
      
    
    parser.add_argument("--report_base_dir", type=str, default="reports/dl_pipeline", help="Base directory to save reports")
    parser.add_argument("--experiment_name", type=str, default=None, help="Specific name for this experiment run (default: timestamped)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")

    # Data params
    parser.add_argument("--binary_classification", action="store_false", default=True, help="Set to False for multi-class (requires model/loss changes)") # Default True
    parser.add_argument("--stratify_split", action="store_false", default=True, help="Use stratified splitting for train/test/holdout") # Default True
    parser.add_argument("--test_split_ratio", type=float, default=0.15, help="Proportion of train_val data to use for testing final model (0 to 1)")
    parser.add_argument("--holdout_split_ratio", type=float, default=0.15, help="Proportion of total data for holdout set (0 to 1). If 0, no holdout set is created.")
    parser.add_argument("--resampling_method", type=str, choices=['SMOTEENN', 'SMOTETomek'], default=None, help="Choose resampling technique for training data.")

    # Model choice
    parser.add_argument("--models_to_run", nargs='+', default=['CNN', 'Transformer', 'CNNTransformer_sequential', 'CNNTransformer_parallel'], choices=['CNN', 'Transformer', 'CNNTransformer_sequential', 'CNNTransformer_parallel'], help="List of models to run")
    parser.add_argument("--select_features", nargs='+', type=int, default=None, help="Space-separated list of feature group indices (0-8) to use. Default is None, which uses all 9 groups. E.g., --select_features 0 5 8")

    # Shared Model Hyperparameters (can be overridden by tuning)
    parser.add_argument("--dropout_rate", type=float, default=0.3, help="Dropout rate for models")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer")

    # CNN specific
    parser.add_argument("--cnn_units", type=int, default=64, help="Number of units in CNN layers")
    # Transformer specific
    parser.add_argument("--transformer_dim", type=int, default=128, help="Dimension of transformer model (d_model)")
    parser.add_argument("--transformer_heads", type=int, default=8, help="Number of heads in transformer multi-head attention")
    parser.add_argument("--transformer_layers", type=int, default=3, help="Number of transformer encoder layers")

    # Training params
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and evaluation")
    parser.add_argument("--epochs_training", type=int, default=30, help="Number of epochs for main model training (CV and final)") # Reduced from 100 for speed
    parser.add_argument("--cv_splits_training", type=int, default=3, help="Number of K-fold CV splits for main training")

    # Hyperparameter Tuning params
    parser.add_argument("--tune_hyperparameters", action="store_true", help="Set this flag to perform hyperparameter tuning before main training (default: off)")
    parser.add_argument("--epochs_tuning", type=int, default=30, help="Number of epochs for models during tuning (per fold)") # Small for faster tuning
    parser.add_argument("--cv_splits_tuning", type=int, default=3, help="Number of K-fold CV splits for tuning")

    # SHAP params
    parser.add_argument("--shap_num_samples", type=int, default=10, help="Number of samples for SHAP analysis background/explanation")


    parsed_args = parser.parse_args()

    # Ensure transformer_dim is divisible by transformer_heads if Transformer is run
    if 'Transformer' in parsed_args.models_to_run or 'CNNTransformer_sequential' in parsed_args.models_to_run or 'CNNTransformer_parallel' in parsed_args.models_to_run:
        if parsed_args.transformer_dim % parsed_args.transformer_heads != 0:
            # Adjust dim to be divisible by heads, or raise error
            original_dim = parsed_args.transformer_dim
            parsed_args.transformer_dim = (original_dim // parsed_args.transformer_heads) * parsed_args.transformer_heads
            if parsed_args.transformer_dim == 0 and original_dim > 0 : # handle case where dim < heads
                parsed_args.transformer_dim = parsed_args.transformer_heads
            logger.warning(f"Transformer dim ({original_dim}) not divisible by heads ({parsed_args.transformer_heads}). Adjusted dim to {parsed_args.transformer_dim}.")
            if parsed_args.transformer_dim == 0 and original_dim >0 :
                 raise ValueError(f"Cannot run transformer with dim {original_dim} and heads {parsed_args.transformer_heads}. Adjusted dim is 0.")


    main(parsed_args)