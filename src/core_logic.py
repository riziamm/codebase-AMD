import pandas as pd
import numpy as np
# import pip
# pip.main(['install','seaborn'])
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shap
from pathlib import Path
import traceback
import signal
import time
import platform 
from sklearn.neighbors import KNeighborsClassifier
import torch
import random
import pickle
from datetime import datetime
from collections import Counter
import csv
import json
import xgboost as xgb
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, LeaveOneOut, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler, RobustScaler,PolynomialFeatures, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier,StackingClassifier
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline 
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.base import clone
from sklearn.feature_selection import SelectFromModel, f_classif
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.naive_bayes import GaussianNB, ComplementNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import matplotlib.patches as patches 
import matplotlib.colors as colors
import math
import logging
# Configure a basic logger for demonstration
logging.basicConfig(level=logging.INFO)
logging.getLogger('shap').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# Set random seeds for reproducibility
def set_seeds(seed=42):
    """
    Set random seeds for reproducibility across all libraries
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed) if torch.cuda.is_available() else None
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
   
def sort_feature_values(X, num_features, values_per_feature=20, ascending_features=None, descending_features=None, unsorted_features=None):
    """
    Sort values within each feature set with flexible ordering options.
    """
    # Make a copy to avoid modifying the original array
    X_copy = X.copy()
    
    # Calculate actual number of features from the data shape
    n_samples = X_copy.shape[0]
    total_columns = X_copy.shape[1]
    
    
    # Use the num_features passed as argument. Do NOT auto-detect/overwrite it here
    # when feature_indices might have been used upstream.
    actual_num_features = num_features
    print(f"Debug - Using num_features passed to function: {actual_num_features}")

    # Optional: Keep the warning check, but don't change actual_num_features
    if total_columns % values_per_feature != 0:
        print(f"Warning - Input data columns ({total_columns}) not divisible by values_per_feature ({values_per_feature})")
    elif total_columns != actual_num_features * values_per_feature:
         print(f"Warning - Input data columns ({total_columns}) do not match expected columns for {actual_num_features} features ({actual_num_features * values_per_feature})")

    # --- END MODIFICATION ---
    
    expected_elements = n_samples * actual_num_features * values_per_feature
    actual_elements = X_copy.size
    
    # Debug output
    print(f"Debug - Sorting dimensions: samples={n_samples}, features={actual_num_features}, values_per_feature={values_per_feature}") # Uses actual_num_features
    print(f"Debug - Expected elements: {expected_elements}, Actual elements: {actual_elements}")

    
    # If dimensions don't match, we can't reshape properly
    if expected_elements != actual_elements:
        raise ValueError(f"Cannot reshape array - dimensions mismatch. Expected {expected_elements} elements "
                       f"for shape ({n_samples}, {actual_num_features}, {values_per_feature}), but got {actual_elements} elements.")
    
    # Reshape to separate features
    try:
        X_reshaped = X_copy.reshape(n_samples, actual_num_features, values_per_feature)
    except ValueError as e:
        print(f"Reshape error: {e}")
        raise ValueError(f"Failed to reshape array with shape {X_copy.shape} to "
                       f"({n_samples}, {actual_num_features}, {values_per_feature}): {str(e)}")
    
    # If all are None, return the original array (no sorting)
    if ascending_features is None and descending_features is None and unsorted_features is None:
        return X_copy
    
    # Initialize the sets of features to sort
    if unsorted_features is None:
        unsorted_features = []
    
    # Determine which features to sort in which direction
    all_features = set(range(actual_num_features))
    unsorted_set = set(unsorted_features)
    sortable_features = list(all_features - unsorted_set)
    
    if ascending_features is None and descending_features is None:
        # Default: sort all non-unsorted features in ascending order
        ascending_features = sortable_features
        descending_features = []
    elif ascending_features is None:
        # Sort remaining sortable features in ascending order
        descending_set = set(descending_features)
        ascending_features = list(all_features - descending_set - unsorted_set)
    elif descending_features is None:
        # Sort remaining sortable features in descending order
        ascending_set = set(ascending_features)
        descending_features = list(all_features - ascending_set - unsorted_set)
    
    # Verify that no feature appears in multiple lists
    asc_set = set(ascending_features)
    desc_set = set(descending_features)
    unsort_set = set(unsorted_features)
    
    # Filter out indices that are out of range
    asc_set = {idx for idx in asc_set if idx < actual_num_features}
    desc_set = {idx for idx in desc_set if idx < actual_num_features}
    unsort_set = {idx for idx in unsort_set if idx < actual_num_features}
    
    # Check for overlaps
    overlaps = []
    if len(asc_set.intersection(desc_set)) > 0:
        overlaps.append(f"Features {asc_set.intersection(desc_set)} appear in both ascending and descending lists")
    if len(asc_set.intersection(unsort_set)) > 0:
        overlaps.append(f"Features {asc_set.intersection(unsort_set)} appear in both ascending and unsorted lists")
    if len(desc_set.intersection(unsort_set)) > 0:
        overlaps.append(f"Features {desc_set.intersection(unsort_set)} appear in both descending and unsorted lists")
    
    if overlaps:
        raise ValueError("\n".join(overlaps))
    
    # Sort features
    for i in range(X_reshaped.shape[0]):  # For each instance
        # Sort ascending features
        for j in asc_set:
            if j < num_features:  # Safety check
                X_reshaped[i, j, :] = np.sort(X_reshaped[i, j, :])
        
        # Sort descending features
        for j in desc_set:
            if j < num_features:  # Safety check
                X_reshaped[i, j, :] = np.sort(X_reshaped[i, j, :])[::-1]
                
        # Unsorted features are left as is
    
    # Reshape back to original shape
    return X_reshaped.reshape(X_copy.shape[0], -1)

# prepare_data function accepts sort_features as a string parameter
def prepare_data(df, num_features=9, values_per_feature=20,
                 normalization='standard', is_binary=True,
                 preserve_zones=True, feature_indices=None, sort_features='none', transform_features=False):
    """
    Prepare data for modeling
    """
    print(f"DEBUG: Inside prepare_data. feature_indices = {feature_indices}, num_features (default/passed) = {num_features}")
    print(f"DEBUG: Input df shape: {df.shape}") # Check input shape early

    # --- Determine actual_num_features FIRST ---
    if feature_indices is not None:
        # Ensure feature_indices is a list-like structure
        if not hasattr(feature_indices, '__iter__') or isinstance(feature_indices, (str, bytes)):
             raise TypeError(f"feature_indices must be an iterable (like a list or tuple), not {type(feature_indices)}")
        actual_num_features = len(feature_indices)
        print(f"DEBUG: Using feature_indices. actual_num_features set to {actual_num_features}")
    else:
        # If no specific indices, use the default/passed num_features
        # This might be adjusted later based on actual data columns found
        actual_num_features = num_features
        print(f"DEBUG: No feature_indices provided. actual_num_features initially set to {actual_num_features}")


    # Extract features based on feature_indices if provided
    if feature_indices is not None:
        # Use only specified features
        print(f"Using selected features: {feature_indices}")

        if preserve_zones:
            selected_X_parts = []
            # Determine the number of columns available for features 
            num_data_cols = df.shape[1] - 1
            print(f"DEBUG: Total columns in DataFrame: {df.shape[1]}. Feature columns assumed: {num_data_cols}")

            for i in feature_indices:
                # Check if index is valid type
                if not isinstance(i, int) or i < 0:
                    raise TypeError(f"Invalid feature index type or value: {i}. Indices must be non-negative integers.")

                start_col = i * values_per_feature
                end_col = start_col + values_per_feature # End index is exclusive
                print(f"DEBUG: Attempting to select columns {start_col} to {end_col-1} for feature index {i}")

                # Check if indices are valid for the DataFrame feature columns
                if start_col >= num_data_cols or end_col > num_data_cols:
                     raise IndexError(f"Feature index {i} is out of bounds. Cannot select columns {start_col}-{end_col-1}. Input data only has {num_data_cols} feature columns.")

                # Select columns using .iloc for position-based indexing
                try:
                    selected_X_parts.append(df.iloc[:, start_col:end_col].values)
                except Exception as e:
                    print(f"ERROR selecting columns {start_col}:{end_col} for index {i}: {e}")
                    raise

            if not selected_X_parts:
                 raise ValueError("No feature columns were selected based on feature_indices. Check indices and data structure.")

            # Horizontally stack the selected parts
            try:
                X = np.hstack(selected_X_parts)
            except Exception as e:
                 print(f"ERROR during np.hstack: {e}")
                 # Print shapes of parts for debugging
                 for idx, part in enumerate(selected_X_parts):
                     print(f"Shape of part {idx}: {part.shape}")
                 raise
            print(f"DEBUG: Shape of X after selecting features via feature_indices: {X.shape}")
           
        else:
            # Select columns for specified features (flattened - less common)
            selected_columns = []
            num_data_cols = df.shape[1] - 1
            for idx in feature_indices:
                if not isinstance(idx, int) or idx < 0:
                    raise TypeError(f"Invalid feature index type or value: {idx}.")
                start_col = idx * values_per_feature
                end_col = start_col + values_per_feature
                if start_col >= num_data_cols or end_col > num_data_cols:
                     raise IndexError(f"Feature index {idx} is out of bounds when selecting columns {start_col}-{end_col-1}.")
                selected_columns.extend(list(range(start_col, end_col)))

            if not selected_columns:
                 raise ValueError("No columns selected for flattened features.")
            X = df.iloc[:, selected_columns].values
            print(f"DEBUG: Shape of X (flattened selection): {X.shape}")


    else:
        # Use all available features (up to num_features*values_per_feature or available columns)
        print(f"DEBUG: Using all available features up to {actual_num_features} specified.")
        max_expected_cols = actual_num_features * values_per_feature
        num_data_cols = df.shape[1] - 1 # Exclude target

        # Determine actual columns to select
        cols_to_select = min(max_expected_cols, num_data_cols)
        if cols_to_select <= 0:
             raise ValueError("No feature columns available to select.")
        print(f"DEBUG: Selecting first {cols_to_select} columns as features.")

        X = df.iloc[:, :cols_to_select].values

        # IMPORTANT: Recalculate actual_num_features based on the columns ACTUALLY selected
        if values_per_feature > 0:
            if X.shape[1] % values_per_feature == 0:
                 actual_num_features = X.shape[1] // values_per_feature
                 print(f"DEBUG: Recalculated actual_num_features based on selected columns: {actual_num_features}")
            else:
                 print(f"WARNING: Selected columns ({X.shape[1]}) not perfectly divisible by values_per_feature ({values_per_feature}). Feature structure might be incorrect.")
                 # Keep the initially assumed actual_num_features or decide how to handle
                 # For safety, might be better to raise error if structure is expected:
                 # raise ValueError(f"Selected columns ({X.shape[1]}) not divisible by values_per_feature ({values_per_feature})")
        else:
            print("WARNING: values_per_feature is zero or negative, cannot determine features from columns.")
            # Keep actual_num_features as initially assumed or handle error
        print(f"DEBUG: Shape of X using all features: {X.shape}")


    # --- Target Variable ---
    # Extract target variable (assuming it's the LAST column)
    try:
        y = df.iloc[:, -1].values
    except IndexError:
         raise IndexError("Could not extract target variable from the last column. DataFrame might be empty or have unexpected structure.")
    print(f"DEBUG: Shape of y: {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch between number of samples in X ({X.shape[0]}) and y ({y.shape[0]})")


    # --- Label Encoding & Binary Conversion ---
    le = LabelEncoder()
    try:
        y = le.fit_transform(y)
    except Exception as e:
         print(f"Error during label encoding: {e}")
         raise

    if is_binary:
        y_binary = np.where(y == 0, 0, 1)  # Class 0 vs rest
        y = y_binary

    print("Class distribution in y:", Counter(y))

    # --- Apply Sorting ---
    # This section assumes sort_feature_values is correctly defined elsewhere
    # and handles the reshaping and sorting based on the num_features it receives.
    if sort_features != 'none':
        print(f"DEBUG: Applying sorting. Sort config: {sort_features}")
        print(f"DEBUG: Value of actual_num_features JUST BEFORE calling sort_feature_values: {actual_num_features}")
        print(f"DEBUG: Shape of X JUST BEFORE calling sort_feature_values: {X.shape}")

        # Verify X shape consistency before calling sort
        expected_cols_for_sort = actual_num_features * values_per_feature
        if X.shape[1] != expected_cols_for_sort:
             raise ValueError(f"CRITICAL: Shape of X ({X.shape}) is inconsistent with {actual_num_features} features and {values_per_feature} values/feature (expected {expected_cols_for_sort} columns) before calling sort_feature_values.")

        ascending_features_to_pass = []
        descending_features_to_pass = []
        unsorted_features_to_pass = []

        if sort_features == 'ascend_all':
            ascending_features_to_pass = list(range(actual_num_features))
        elif sort_features == 'descend_all':
            descending_features_to_pass = list(range(actual_num_features))
        elif sort_features == 'custom':
             # Default custom: sort first two asc, third desc (relative indices)
            ascending_features_to_pass = [0, 1] if actual_num_features > 1 else ([0] if actual_num_features == 1 else [])
            descending_features_to_pass = [2] if actual_num_features > 2 else []
        elif isinstance(sort_features, dict):
            # Map original indices from dict to local indices
            original_ascending = sort_features.get('ascending', [])
            original_descending = sort_features.get('descending', [])
            original_unsorted = sort_features.get('unsorted', [])

            if feature_indices is not None:
                # Create mapping from original to local indices
                feature_map = {orig_idx: local_idx for local_idx, orig_idx in enumerate(feature_indices)}
                print(f"DEBUG: Feature index map: {feature_map}")

                ascending_features_to_pass = [feature_map[idx] for idx in original_ascending if idx in feature_map]
                descending_features_to_pass = [feature_map[idx] for idx in original_descending if idx in feature_map]
                unsorted_features_to_pass = [feature_map[idx] for idx in original_unsorted if idx in feature_map]

                print(f"DEBUG: Mapped sorting indices - Asc: {ascending_features_to_pass}, Desc: {descending_features_to_pass}, Unsorted: {unsorted_features_to_pass}")
            else:
                 # If using all features, the indices in the dict are already local
                 ascending_features_to_pass = [idx for idx in original_ascending if idx < actual_num_features]
                 descending_features_to_pass = [idx for idx in original_descending if idx < actual_num_features]
                 unsorted_features_to_pass = [idx for idx in original_unsorted if idx < actual_num_features]
                 print(f"DEBUG: Using dictionary indices directly (no feature_indices) - Asc: {ascending_features_to_pass}, Desc: {descending_features_to_pass}, Unsorted: {unsorted_features_to_pass}")
        else:
             print(f"WARNING: Unknown sort_features type or value: {sort_features}. No sorting applied.")


        # Call the sorting function
        try:
             # Ensure sort_feature_values is defined and imported
             X = sort_feature_values(X, num_features=actual_num_features,
                                     values_per_feature=values_per_feature,
                                     ascending_features=ascending_features_to_pass,
                                     descending_features=descending_features_to_pass,
                                     unsorted_features=unsorted_features_to_pass)
             print(f"DEBUG: Shape of X after sorting: {X.shape}")
        except NameError:
              print("ERROR: sort_feature_values function not found. Please ensure it is defined and imported.")
              raise
        except Exception as e:
              print(f"ERROR during sort_feature_values call: {e}")
              raise # Re-raise the original error after printing context

    else:
         print("No sorting applied.")


    # --- Feature Name Generation ---
    # Base names - adjust if your features represent different things
    base_metric_names = ['mean', 'median', 'std', 'iqr', 'idr', 'skew', 'kurt', 'Del', 'Amp']
    actual_group_metrics = []
    
    # Check if actual_num_features matches the expected number based on metrics
    if feature_indices is not None: 
         # Use metric names corresponding to selected feature indices
         feature_names = [f'{base_metric_names[feat_idx]}_Z{zone_idx+1}' for feat_idx in feature_indices for zone_idx in range(values_per_feature)] 
         actual_group_metrics = [base_metric_names[feat_idx] for feat_idx in feature_indices] #
    elif actual_num_features != len(base_metric_names): 
         print(f"WARNING: actual_num_features ({actual_num_features}) doesn't match number of base_metric_names ({len(base_metric_names)}). Feature names might be inaccurate.")
         if abs(actual_num_features - len(base_metric_names)) > 1: # Use generic names if mismatch is significant
             feature_names = [f'F{feat_idx}_Z{zone_idx+1}' for feat_idx in range(actual_num_features) for zone_idx in range(values_per_feature)] 
             actual_group_metrics = [f'F{feat_idx}' for feat_idx in range(actual_num_features)] 
         else: # Assume metrics list might be slightly off, use actual_num_features
             feature_names = [f'{base_metric_names[feat_idx]}_Z{zone_idx+1}' for feat_idx in range(actual_num_features) for zone_idx in range(values_per_feature)] 
             actual_group_metrics = base_metric_names[:actual_num_features] #
    else: # for feature_indices None, use actual_num_features if matches len(base_metric_names)
         feature_names = [f'{base_metric_names[feat_idx]}_Z{zone_idx+1}' for feat_idx in range(actual_num_features) for zone_idx in range(values_per_feature)] 
         actual_group_metrics = base_metric_names[:actual_num_features] #

    print(f"DEBUG: Generated {len(feature_names)} feature names. First few: {feature_names[:5]}...") 
    print(f"DEBUG: Actual group metrics used: {actual_group_metrics}")


    # --- Feature Transformation (Optional) ---
    # create_feature_transformations handles fitting and transforming correctly
    if transform_features:
        print(f"Applying feature transformations for {actual_num_features} features...")
        try:
             X, feature_names = create_feature_transformations_fit(
                 X, feature_names, actual_num_features, values_per_feature
             )
             print(f"DEBUG: Shape of X after transformation: {X.shape}")
             print(f"DEBUG: Number of feature names after transformation: {len(feature_names)}")
        except NameError:
             print("ERROR: create_feature_transformations function not found. Skipping.")
        except Exception as e:
             print(f"Error during feature transformation: {e}")

    # --- Normalization ---
    if normalization != 'none':
        print(f"Applying {normalization} normalization...")
        if normalization == 'standard':
            scaler = StandardScaler()
        elif normalization == 'minmax':
            scaler = MinMaxScaler()
        elif normalization == 'robust':
            scaler = RobustScaler()
        else:
            print(f"WARNING: Unknown normalization type '{normalization}'. Skipping normalization.")
            scaler = None # # Indicate no scaling if type is 'unknown' but not 'none'

        if scaler:
             try:
                 X = scaler.fit_transform(X)
                 print(f"DEBUG: Shape of X after normalization: {X.shape}")
             except Exception as e:
                 print(f"Error during normalization: {e}")
                 raise
    else:
         print("No normalization applied.")
         scaler = None # Initialize scaler to None when normalization is 'none'

    return X, y, le, scaler, feature_names, actual_group_metrics


def preprocess_holdout_data(df_holdout, config, saved_le, saved_scaler=None):
    """
    Preprocesses holdout data using parameters and objects learned from training.

    Args:
        df_holdout (pd.DataFrame): The raw holdout data loaded into a DataFrame.
        config (dict): The configuration dictionary FROM THE TRAINING RUN.
        saved_le (LabelEncoder): The LabelEncoder fitted on the training target.
        saved_scaler (Scaler, optional): The Scaler fitted on the training features. Defaults to None.

    Returns:
        tuple: X_holdout_processed, y_holdout_processed, feature_names
               Returns (None, None, None) if processing fails.
    """
    print("Preprocessing holdout data using configuration and objects from training run...")
    try:
        # --- Extract relevant config parameters from the loaded config ---
        target_col = config.get('target_variable', 'Y') # Ensure target_col is in config or use default
        is_binary = config.get('is_binary', True)
        preserve_zones = config.get('preserve_zones', True) # Typically True if trained that way
        feature_indices = config.get('feature_indices', None) # Use indices from training
        num_features_trained = config.get('num_features', 9) # Get num_features used in training
        values_per_feature = config.get('values_per_feature', 20) # Get values_per_feature from config
        sort_features_config = config.get('sort_features', 'none') # Get sorting config from training
        transform_features_config = config.get('transform_features', False) # Check if transformation was done in training
        normalization_config = config.get('normalization', 'none') # Get normalization from training

        print(f"DEBUG (Holdout): Using training config - is_binary={is_binary}, preserve_zones={preserve_zones}, feature_indices={feature_indices}, num_features_trained={num_features_trained}, sort={sort_features_config}, transform={transform_features_config}, normalization={normalization_config}")

        if transform_features_config:
             print("WARNING: Holdout preprocessing currently doesn't support reapplying 'transform_features'. Features will be used without this specific transformation.")
             # If you implement saved transformation parameters, load and apply them here.

        # --- Determine Feature Structure Based on Training Config ---
        if feature_indices is not None:
             # Use exactly the features specified during training
             actual_num_features = len(feature_indices)
             print(f"DEBUG (Holdout): Selecting features based on training indices: {feature_indices}. Expecting {actual_num_features} features.")
             selected_X_parts = []
             num_data_cols = df_holdout.shape[1] - 1 # Exclude target
             for i in feature_indices:
                 if not isinstance(i, int) or i < 0: raise TypeError(f"Invalid feature index in config: {i}")
                 start_col = i * values_per_feature
                 end_col = start_col + values_per_feature
                 if start_col >= num_data_cols or end_col > num_data_cols:
                     raise IndexError(f"Holdout data: Feature index {i} from training config is out of bounds for holdout data shape {df_holdout.shape}.")
                 selected_X_parts.append(df_holdout.iloc[:, start_col:end_col].values)
             if not selected_X_parts: raise ValueError("Holdout data: No features selected based on training indices.")
             X_holdout = np.hstack(selected_X_parts)
        else:
             # Use the number of features determined during training
             actual_num_features = num_features_trained
             print(f"DEBUG (Holdout): Selecting features based on num_features_trained={actual_num_features}.")
             max_expected_cols = actual_num_features * values_per_feature
             num_data_cols = df_holdout.shape[1] - 1
             cols_to_select = min(max_expected_cols, num_data_cols)
             if cols_to_select <= 0: raise ValueError("Holdout data: No feature columns available.")
             if cols_to_select != max_expected_cols:
                 print(f"WARNING (Holdout): Holdout data has fewer columns ({num_data_cols}) than expected ({max_expected_cols}) based on training config. Selecting first {cols_to_select}.")
             X_holdout = df_holdout.iloc[:, :cols_to_select].values
             # Verify the number of features derived matches training
             if values_per_feature > 0 and X_holdout.shape[1] % values_per_feature == 0:
                 derived_num_features = X_holdout.shape[1] // values_per_feature
                 if derived_num_features != actual_num_features:
                     print(f"WARNING (Holdout): Derived num features ({derived_num_features}) differs from training config ({actual_num_features}). Using derived.")
                     actual_num_features = derived_num_features # Adjust if structure differs
             elif values_per_feature > 0:
                 raise ValueError(f"Holdout data selected columns ({X_holdout.shape[1]}) not divisible by values_per_feature ({values_per_feature}).")


        print(f"DEBUG (Holdout): X shape after feature selection: {X_holdout.shape}")

        # --- Target Variable ---
        if target_col not in df_holdout.columns:
            raise ValueError(f"Holdout data ERROR: Target column '{target_col}' not found.")
        y_holdout = df_holdout[target_col].values

        # --- Apply Saved LabelEncoder & Handle Unseen Labels ---
        if saved_le is None: raise ValueError("LabelEncoder object is required.")
        seen_labels = saved_le.classes_
        mask_seen = np.isin(y_holdout, seen_labels)
        if not mask_seen.all():
            print(f"Holdout data WARNING: Found {sum(~mask_seen)} samples with labels unseen during training. These samples will be REMOVED.")
            X_holdout = X_holdout[mask_seen]
            y_holdout = y_holdout[mask_seen] # Filter original labels first
            if X_holdout.shape[0] == 0:
                print("Holdout data ERROR: No samples remaining after filtering unseen labels.")
                return None, None, None
        y_holdout_processed = saved_le.transform(y_holdout) # Transform the filtered labels

        print(f"DEBUG (Holdout): Shape after filtering unseen labels - X:{X_holdout.shape}, y:{y_holdout_processed.shape}")

        # --- Binary Conversion (Based on Training Config) ---
        if is_binary:
            # Use the same positive class logic as in prepare_data if needed, based on config
            y_holdout_processed = np.where(y_holdout_processed == 0, 0, 1) # Assuming class 0 vs rest

        # --- Feature Sorting (Based on Training Config) ---
        if sort_features_config != 'none':
            print(f"DEBUG (Holdout): Applying sorting based on training config: {sort_features_config}")
            # (Copy the sorting logic block from prepare_data here, ensuring it uses
            # actual_num_features, values_per_feature, sort_features_config, and feature_indices
            # correctly to determine ascending/descending/unsorted features for the call)
            # --- Start copy sorting logic ---
            ascending_features_to_pass = []
            descending_features_to_pass = []
            unsorted_features_to_pass = []

            if sort_features_config == 'ascend_all':
                ascending_features_to_pass = list(range(actual_num_features))
            elif sort_features_config == 'descend_all':
                descending_features_to_pass = list(range(actual_num_features))
            elif sort_features_config == 'custom':
                ascending_features_to_pass = [0, 1] if actual_num_features > 1 else ([0] if actual_num_features == 1 else [])
                descending_features_to_pass = [2] if actual_num_features > 2 else []
            elif isinstance(sort_features_config, dict):
                original_ascending = sort_features_config.get('ascending', [])
                original_descending = sort_features_config.get('descending', [])
                original_unsorted = sort_features_config.get('unsorted', [])
                if feature_indices is not None:
                    feature_map = {orig_idx: local_idx for local_idx, orig_idx in enumerate(feature_indices)}
                    ascending_features_to_pass = [feature_map[idx] for idx in original_ascending if idx in feature_map]
                    descending_features_to_pass = [feature_map[idx] for idx in original_descending if idx in feature_map]
                    unsorted_features_to_pass = [feature_map[idx] for idx in original_unsorted if idx in feature_map]
                else:
                     ascending_features_to_pass = [idx for idx in original_ascending if idx < actual_num_features]
                     descending_features_to_pass = [idx for idx in original_descending if idx < actual_num_features]
                     unsorted_features_to_pass = [idx for idx in original_unsorted if idx < actual_num_features]
            else:
                 print(f"WARNING (Holdout): Unknown sort_features type: {sort_features_config}. No sorting applied.")

            if ascending_features_to_pass or descending_features_to_pass or unsorted_features_to_pass:
                 try:
                     # Ensure sort_feature_values is accessible
                     X_holdout = sort_feature_values(X_holdout, num_features=actual_num_features,
                                                     values_per_feature=values_per_feature,
                                                     ascending_features=ascending_features_to_pass,
                                                     descending_features=descending_features_to_pass,
                                                     unsorted_features=unsorted_features_to_pass)
                     print(f"DEBUG (Holdout): Shape after sorting: {X_holdout.shape}")
                 except Exception as e:
                      print(f"ERROR during holdout sort_feature_values call: {e}")
                      raise
            # --- End copy sorting logic ---
        else:
            print("DEBUG (Holdout): Skipping sorting based on training config.")


        # --- Feature Transformation (Based on Training Config - Placeholder) ---
        if transform_features_config:
            print("DEBUG (Holdout): Applying transformations (if implementation exists)...")
            # ADD LOGIC HERE: Load transformation parameters/objects saved during training
            # and apply the *transform* method to X_holdout.
            # Example:
            # loaded_poly_transformer = load_pickle(...)
            # X_holdout = loaded_poly_transformer.transform(X_holdout)
            pass # Replace with actual transformation code if needed


        # --- Apply Saved Normalization (Based on Training Config) ---
        X_holdout_processed = X_holdout # Default if no scaling needed/done
        if normalization_config != 'none':
            if saved_scaler:
                print(f"DEBUG (Holdout): Applying saved '{normalization_config}' scaler...")
                X_holdout_processed = saved_scaler.transform(X_holdout) # Use TRANSFORM only
            else:
                # This case should ideally not happen if config says normalize but scaler wasn't saved/loaded
                print(f"WARNING (Holdout): Training config specified '{normalization_config}' but no scaler provided. Using unscaled data.")
        else:
             print("DEBUG (Holdout): Skipping normalization based on training config.")


        # --- Handle NaNs (Impute based on holdout data - less ideal but practical) ---
        if not np.all(np.isfinite(X_holdout_processed)):
             print("WARNING (Holdout): Non-finite values detected after processing. Imputing with mean.")
             # Consider saving training means for imputation if possible, otherwise use holdout mean
             imputer = SimpleImputer(strategy='mean')
             X_holdout_processed = imputer.fit_transform(X_holdout_processed)


        # --- Generate Feature Names (Consistent with Training) ---
        # Use the same logic as in prepare_data, based on actual_num_features and feature_indices
        metrics_basic = ['mean', 'median', 'std', 'iqr', 'idr', 'skew', 'kurt', 'Del', 'Amp']
        num_metrics = len(metrics_basic)
        if feature_indices:
             feature_names = [f'{metrics_basic[idx]}_Z{z+1}' for idx in feature_indices for z in range(values_per_feature)]
        else:
             feature_names = [f'{metrics_basic[m_idx]}_Z{z+1}' for m_idx in range(min(actual_num_features, num_metrics)) for z in range(values_per_feature)]
             feature_names.extend([f'F{m_idx}_Z{z+1}' for m_idx in range(num_metrics, actual_num_features) for z in range(values_per_feature)])

        # Add names for transformed features if transform_features_config was True AND implemented
        if transform_features_config:
             # Assuming create_feature_transformations would have returned names
             # This needs adjustment based on how you save/load transformed names
             # Example: loaded_transform_names = load_pickle(...)
             # feature_names = loaded_transform_names
             print("WARNING (Holdout): Feature names may not reflect transformations applied during training.")


        # Final check for consistency
        if len(feature_names) != X_holdout_processed.shape[1]:
              print(f"WARNING (Holdout): Generated feature names ({len(feature_names)}) mismatch processed columns ({X_holdout_processed.shape[1]}). Using generic names.")
              feature_names = [f'feature_{i}' for i in range(X_holdout_processed.shape[1])]


        print("Holdout data preprocessing complete.")
        return X_holdout_processed, y_holdout_processed, feature_names

    except Exception as e:
        print(f"ERROR during holdout preprocessing: {e}")
        traceback.print_exc()
        return None, None, None


# Plot class distributions  
def plot_class_distribution(y, le, title):
    """Plot distribution of classes"""
    plt.figure(figsize=(6, 3)) 
    
    # Convert y to numeric type to avoid categorical warning
    y_numeric = y.astype(int)
    
    # Create plot with numeric classes
    class_counts = np.bincount(y_numeric)
    classes = np.arange(len(class_counts))
    
    # Use color instead of palette (matplotlib vs seaborn parameter)
    bars = plt.bar(classes, class_counts, color='skyblue')
    
    # Add count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom')
    
    # Try to use original labels if available
    try:
        if hasattr(le, 'classes_'):
            original_labels = le.classes_
            plt.xticks(classes, original_labels)
        else:
            plt.xticks(classes, [f'Class {i}' for i in classes])
    except:
        plt.xticks(classes, [f'Class {i}' for i in classes])
    
    plt.title(title)
    plt.xlabel('Class Labels')
    plt.ylabel('Count')
    plt.tight_layout()
    # plt.close()

# Calculate Minkowski distances
def calculate_minkowski_distances(X, p=2, max_samples=1000):
    """
    Calculate Minkowski distances between data points
    
    Args:
        X: Input data
        p: Minkowski distance parameter
        max_samples: Maximum number of samples to use
        
    Returns:
        distances: Array of distances
    """
    distances = []
    # Take a sample if dataset is large
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    for i in range(len(X_sample)):
        for j in range(i + 1, len(X_sample)):
            dist = minkowski(X_sample[i], X_sample[j], p)
            distances.append(dist)
    return np.array(distances)

# Function to analyze Minkowski distances
def analyze_minkowski(X_train, p_values=None):
    """
    Analyze Minkowski distances for given p values
    
    Args:
        X_train: Training data
        p_values: List of p values for Minkowski distance (default: [1,2,3])
    """
    if p_values is None:
        p_values = [1, 2, 3]
    
    fig, axes = plt.subplots(1, len(p_values), figsize=(4*len(p_values), 3))  # Reduced size
    if len(p_values) == 1:
        axes = [axes]
    
    for i, p in enumerate(p_values):
        distances = calculate_minkowski_distances(X_train, p=p)
        axes[i].hist(distances, bins=30)
        axes[i].set_title(f'Minkowski Distances (p={p})')
        axes[i].set_xlabel('Distance')
        axes[i].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    plt.close()

# Model training and evaluation function
def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, le):
    """
    Train and evaluate a model
    
    Args:
        model: Model to train
        X_train, X_test, y_train, y_test: Train/test data
        model_name: Name of the model
        le: Label encoder
        
    Returns:
        model: Trained model
        y_pred: Predictions
        roc_auc: ROC AUC score (float or None)
        avg_precision: Average Precision score (float or None)
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    roc_auc = None
    avg_precision = None
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted',zero_division=0)
    
    # Calculate ROC AUC for binary classification
    # if len(np.unique(y_test)) == 2:
    #     # Check if model can predict probabilities
    #     if hasattr(model, "predict_proba"):
    #         y_proba = model.predict_proba(X_test)[:, 1]
    #         roc_auc = roc_auc_score(y_test, y_proba)
    #         # Plot ROC curve
    #         fpr, tpr, _ = roc_curve(y_test, y_proba)
    #         plt.figure(figsize=(6, 6))
    #         plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    #         plt.plot([0, 1], [0, 1], 'k--')
    #         plt.xlim([0.0, 1.0])
    #         plt.ylim([0.0, 1.05])
    #         plt.xlabel('False Positive Rate')
    #         plt.ylabel('True Positive Rate')
    #         plt.title(f'ROC Curve - {model_name}')
    #         plt.legend(loc="lower right")
    #         plt.show()
            
    #         # Also add precision-recall curve
    #         precision, recall, _ = precision_recall_curve(y_test, y_proba)
    #         avg_precision = average_precision_score(y_test, y_proba)
    #         plt.figure(figsize=(6, 6))
    #         plt.plot(recall, precision, label=f'AP = {avg_precision:.2f}')
    #         plt.xlabel('Recall')
    #         plt.ylabel('Precision')
    #         plt.title(f'Precision-Recall Curve - {model_name}')
    #         plt.legend(loc="lower left")
    #         plt.show()
            
    #         print(f"ROC AUC: {roc_auc:.4f}")
    #         print(f"Average Precision: {avg_precision:.4f}")
    #     else:
    #         print("Model doesn't support probability predictions for ROC AUC calculation")
    
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        if y_proba.shape[1] == 2: # Binary classification
            y_proba_pos = y_proba[:, 1]
            try:
                roc_auc = roc_auc_score(y_test, y_proba_pos)
                avg_precision = average_precision_score(y_test, y_proba_pos)
                print(f"ROC AUC: {roc_auc:.4f}")
                print(f"Average Precision: {avg_precision:.4f}")

                # Plot ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_proba_pos)
                plt.figure(figsize=(6, 6)) # Consider reducing figure size for batch runs or making plotting optional
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve - {model_name}')
                plt.legend(loc="lower right")
                plt.show() # In a script, you might want to savefig and close
                plt.close()

                # Plot precision-recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_proba_pos)
                plt.figure(figsize=(6, 6)) # Similar consideration for figure size/plotting
                plt.plot(recall, precision, label=f'AP = {avg_precision:.2f}')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Curve - {model_name}')
                plt.legend(loc="lower left")
                plt.show() # In a script, you might want to savefig and close
                plt.close()
            except ValueError as e:
                print(f"Could not calculate ROC AUC / AP for {model_name} (binary case): {e}")
        elif y_proba.shape[1] > 2: # Multiclass classification
            try:
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
                # report per-class AP if possible or just ROC AUC.
                print(f"ROC AUC (Macro OvR): {roc_auc:.4f}")
                # Placeholder for multiclass AP, avg_precision = "N/A for multiclass in this basic impl."
            except ValueError as e:
                print(f"Could not calculate ROC AUC for {model_name} (multiclass case): {e}")
    else:
        print(f"Model {model_name} doesn't support probability predictions for ROC AUC/AP calculation")

    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # For multiclass, converting back and forth isn't necessary
    # but keeping for consistency with original code
    try:
        y_test_orig = le.inverse_transform(y_test)
        y_pred_orig = le.inverse_transform(y_pred)
        
        print("\nClassification Report:")
        print(classification_report(y_test_orig, y_pred_orig, zero_division=0))
        
        # Plot confusion matrix
        plt.figure(figsize=(4, 3))  # Reduced size
        ConfusionMatrixDisplay.from_predictions(y_test_orig, y_pred_orig)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.show()
        plt.close()
    except:
        # In case inverse_transform fails (e.g. for binary classification)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Plot confusion matrix
        plt.figure(figsize=(4, 3))  # Reduced size
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.tight_layout()
        plt.show()
        # save_figure(plt.gcf(), "conf_matrix", report_dir)
        plt.close()
    
    return model, y_pred, roc_auc, avg_precision




# Save model results to CSV
def save_results(model_name, accuracy, roc_auc, avg_precision, hyperparams, num_features, data_path, is_binary, preserve_zones, sort_features, normalization, sampling_method, y_true, y_pred):
    """
    Save model results to CSV
    
    Args:
        Various parameters describing the experiment
        y_true, y_pred: True and predicted labels
    """
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    
    # Calculate F1 score based on whether it's binary or multiclass
    if len(np.unique(y_true)) == 2:  # Binary
        f1 = report['weighted avg']['f1-score']
    else:  # Multiclass
        f1 = report['macro avg']['f1-score']
    
    feature_groups = num_features // 20
    
    if isinstance(sort_features, str):
        sort_features_str = sort_features  
    elif isinstance(sort_features, dict):
        # Convert dictionary to a meaningful string representation
        parts = []
        if 'ascending' in sort_features and sort_features['ascending']:
            parts.append(f"asc:{','.join(map(str, sort_features['ascending']))}")
        if 'descending' in sort_features and sort_features['descending']:
            parts.append(f"desc:{','.join(map(str, sort_features['descending']))}")
        if 'unsorted' in sort_features and sort_features['unsorted']:
            parts.append(f"unsorted:{','.join(map(str, sort_features['unsorted']))}")
        
        sort_features_str = "|".join(parts) if parts else "custom"
    else:
        # For any other case (including None or unsupported types)
        sort_features_str = str(sort_features)
    
    results = [
        model_name,
        accuracy,
        f1,
        roc_auc if roc_auc is not None else 'N/A', # Add ROC AUC
        avg_precision if avg_precision is not None else 'N/A', # Add Avg Precision
        report['weighted avg']['precision'],
        report['weighted avg']['recall'],
        feature_groups, # num_features,
        str(hyperparams),
        data_path,
        "binary" if is_binary else "multiclass",
        "preserve" if preserve_zones else "flatten",
        # "sorted" if sort_features else "original",
        sort_features_str,  
        normalization,
        sampling_method,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ]
    
    # Determine correct filename
    filename = "model_results_bin.csv" if is_binary else "model_results_mc.csv"
    
    # Check if file exists
    file_exists = os.path.isfile(filename)
    
    # Save results
    with open(filename, "a", newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Model', 'Accuracy', 'F1_Score','ROC_AUC',          
                             'Avg_Precision', 'Precision', 'Recall' 
                             'Num_Features',  'Hyperparameters', 'Data_Path', 
                             'Classification_Type',
                           'Zone_Preservation', 'Feature_Sorting', 'Normalization',
                           'Sampling_Method', 'Timestamp'])
        writer.writerow(results)

# Save the best model
def save_best_model(model, model_name, params):
    """
    Save the best performing model and its parameters
    
    Args:
        model: Model to save
        model_name: Name of the model
        params: Parameters used for the model
    """
    # Create directory if it doesn't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    
    # Generate filename based on model name and parameters
    is_binary = params.get('is_binary', True)
    normalization = params.get('normalization', 'standard')
    sampling = params.get('sampling_method', 'none')
    sort_option = params.get('sort_features', 'none')
    
    filename = f"models/{model_name}_{'binary' if is_binary else 'multiclass'}_{normalization}_{sampling}_{sort_option}.pkl"
    
    # Save model
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    
    # Save parameters
    with open(filename.replace('.pkl', '_params.pkl'), 'wb') as f:
        pickle.dump(params, f)
    
    print(f"Best model ({model_name}) saved to: {filename}")


#------------------------Added--------------------------------
def log_grid_search_results(model_name, best_params, train_score, test_score, gap, report_dir=None):
    """Log grid search results to a file for later analysis"""
    log_entry = {
        "model": model_name,
        "params": best_params,
        "train_score": train_score,
        "test_score": test_score,
        "gap": gap,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create log file name
    log_file = "hyperparameter_tuning_results.json"
    if report_dir:
        log_file = os.path.join(report_dir, log_file)
    
    # Append to existing log or create new one
    try:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
            
        logs.append(log_entry)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
            
        print(f"Logged results for {model_name} to {log_file}")
    except Exception as e:
        print(f"Error logging results: {e}")


def grid_search_model_v3(model, param_grid, X_train, y_train, model_name, cv=3, report_dir=None):
    """
    Perform grid search for hyperparameter tuning with focus on preventing overfitting
    
    Args:
        model: Model to tune
        param_grid: Parameter grid
        X_train, y_train: Training data
        model_name: Name of the model
        cv: Maximum number of cross-validation folds
        
    Returns:
        best_model: Best model
        best_params: Best parameters
    """
    print(f"\nPerforming grid search for {model_name}...")
    print(f"Binary classification: {len(np.unique(y_train)) == 2}")
    print(f"Class distribution in training data: {np.bincount(y_train)}")
    
    # Check if cv is too large for the minority class
    class_counts = np.bincount(y_train)
    min_samples = min(class_counts[class_counts > 0])  # Ignore classes with zero samples
    
    # Determine appropriate number of folds
    original_cv = cv
    if min_samples < 5:  # Very small class
        print(f"Using Leave-One-Out CV due to very small class size: {min_samples}")
        from sklearn.model_selection import LeaveOneOut
        stratified_cv = LeaveOneOut()
    elif min_samples < cv:  # Small but not tiny class
        adjusted_cv = max(2, min(3, min_samples))
        print(f"Adjusting grid search: Minority class has only {min_samples} samples, "
              f"reducing folds from {original_cv} to {adjusted_cv}")
        stratified_cv = StratifiedKFold(n_splits=adjusted_cv, shuffle=True, random_state=42)
    else:  # Class is large enough for normal CV
        print(f"Using standard {cv}-fold cross-validation")
        stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Limit parallel jobs to avoid memory issues with small datasets
    # n_jobs = min(2, os.cpu_count() or 1)
    n_jobs =  1    
    # Print parameter grid for debugging
    print(f"Parameter grid: {param_grid}")
    
    filtered_params = filter_param_grid(param_grid)
    
    
    
    # Ensure the grid search does not use test data information
    grid_search = GridSearchCV(
        model, filtered_params, 
        # param_grid, 
        cv=stratified_cv,  # Use stratified k-fold
        scoring='f1_weighted',
        n_jobs=n_jobs, 
        verbose=1,
        return_train_score=True,  # Add this to check for overfitting
        error_score=0.0  # Return 0 for failed fits instead of raising error
    )
   
    # Try/except block to catch and debug issues
    try:
        print(f"Starting grid search with CV splits: {getattr(stratified_cv, 'n_splits', 'LOO')}")
        grid_search.fit(X_train, y_train)
    
        # Debug overfitting
        # Print CV results to help diagnose overfitting
        means_train = grid_search.cv_results_['mean_train_score']
        means_test = grid_search.cv_results_['mean_test_score']
        stds_test = grid_search.cv_results_['std_test_score']
        
        print(f"Cross-validation results:")
        print(f"{'Parameters':<40} {'Train Score':<12} {'Test Score':<12} {'Gap':<8}")
        for train_mean, test_mean, std, params in zip(means_train, means_test, stds_test, grid_search.cv_results_['params']):
            # Calculate gap between train and test score (potential overfitting indicator)
            gap = train_mean - test_mean
            param_str = str(params)
            if len(param_str) > 35:
                param_str = param_str[:32] + "..."
            print(f"{param_str:<40} {train_mean:.4f} {test_mean:.4f} (±{std:.4f}) {gap:.4f}")
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Check if overfitting is likely (large gap between train and test scores)
        best_idx = grid_search.best_index_
        best_train_score = means_train[best_idx]
        best_test_score = means_test[best_idx]
        gap = best_train_score - best_test_score
        
        if gap > 0.1:  # More than 10% gap indicates potential overfitting
            print(f"Warning: Possible overfitting (10%+ gap). Train score: {best_train_score:.4f}, Test score: {best_test_score:.4f}, Gap: {gap:.4f}")
            
            # Apply regularization if overfitting is severe
            if gap > 0.2:  # More than 20% gap indicates severe overfitting
                print("Severe overfitting detected. Applying stronger regularization...")
                best_model = apply_regularization(model, grid_search.best_params_, strength=1.5)
                best_model.fit(X_train, y_train)
                return best_model, grid_search.best_params_
            else:
                # Moderate overfitting - apply milder regularization
                print("Moderate overfitting detected. Applying mild regularization...")
                best_model = apply_regularization(model, grid_search.best_params_, strength=1.2)
                best_model.fit(X_train, y_train)
                return best_model, grid_search.best_params_
        
        
        if hasattr(grid_search, 'best_score_') and hasattr(grid_search, 'best_params_'):
            # best_idx = grid_search.best_index_
            # best_train_score = means_train[best_idx] if 'means_train' in locals() else 0.0
            # best_test_score = grid_search.best_score_
            # gap = best_train_score - best_test_score
            
            # Log the results
            log_grid_search_results(
                model_name, 
                grid_search.best_params_, 
                best_train_score, 
                best_test_score, 
                gap,
                report_dir
            )
        
        return grid_search.best_estimator_, grid_search.best_params_

        
    except Exception as e:
        print(f"Error during grid search for {model_name}: {str(e)}")
        print("Returning model with conservative parameters")
        return create_conservative_model(model_name), {}

def grid_search_model(model, param_grid, X_train, y_train, model_name, cv=3, report_dir=None, sampling_method='none'): # <-- ADDED
    """
    Perform grid search for hyperparameter tuning with focus on preventing overfitting
    
    Args:
        model: Model to tune
        param_grid: Parameter grid
        X_train, y_train: Training data
        model_name: Name of the model
        cv: Maximum number of cross-validation folds
        
    Returns:
        best_model: Best model
        best_params: Best parameters
    """
    print(f"\nPerforming grid search for {model_name}...")
    print(f"Binary classification: {len(np.unique(y_train)) == 2}")
    print(f"Class distribution in training data: {np.bincount(y_train)}")
    
    # Check if cv is too large for the minority class
    class_counts = np.bincount(y_train)
    min_samples = min(class_counts[class_counts > 0])  # Ignore classes with zero samples
    
    # Determine appropriate number of folds
    original_cv = cv
    if min_samples < 5:  # Very small class
        print(f"Using Leave-One-Out CV due to very small class size: {min_samples}")
        from sklearn.model_selection import LeaveOneOut
        stratified_cv = LeaveOneOut()
    elif min_samples < cv:  # Small but not tiny class
        adjusted_cv = max(2, min(3, min_samples))
        print(f"Adjusting grid search: Minority class has only {min_samples} samples, "
              f"reducing folds from {original_cv} to {adjusted_cv}")
        stratified_cv = StratifiedKFold(n_splits=adjusted_cv, shuffle=True, random_state=42)
    else:  # Class is large enough for normal CV
        print(f"Using standard {cv}-fold cross-validation")
        stratified_cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # Limit parallel jobs to avoid memory issues with small datasets
    # n_jobs = min(2, os.cpu_count() or 1)
    n_jobs =  1
    
    # Print parameter grid for debugging
    print(f"Parameter grid: {param_grid}")
    
    filtered_params = filter_param_grid(param_grid)
    
    print(f"  Sampling Method requested for CV: {sampling_method}") # Log sampling method

    sampler = None
    model_to_tune = clone(model) # Use a clone of the input model
    grid_to_use = filtered_params # Use the filtered parameters by default

    if sampling_method == 'smote':
        # Optional: Add checks for minimum class size if needed, similar to your CV checks
        class_counts_check = np.bincount(y_train)
        min_samples_check = min(class_counts_check[class_counts_check > 0]) if len(class_counts_check[class_counts_check > 0]) > 0 else 0
        unique_classes_check = len(class_counts_check[class_counts_check > 0])

        if unique_classes_check < 2 or min_samples_check < 2:
            print(f"  Warning: Cannot apply SMOTE (need >=2 classes with >=2 samples each). Proceeding without sampling.")
        else:
            # Use k_neighbors logic similar to your run_classification_pipeline if desired
            k_neighbors = min(min_samples_check - 1, 5) # Default k=5, ensure k < min_samples
            if k_neighbors < 1: k_neighbors = 1 # Ensure k is at least 1
            sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
            print(f"    Preparing imblearn pipeline with SMOTE (k={k_neighbors}).")
            
    elif sampling_method == 'random_over': 
        print(f"    Preparing imblearn pipeline with RandomOverSampler.")
        sampler = RandomOverSampler(random_state=42)

    elif sampling_method == 'random_under': 
        print(f"    Preparing imblearn pipeline with RandomUnderSampler.")
        sampler = RandomUnderSampler(random_state=42)

    # Add elif for other samplers if you implement them ('random_over', 'random_under')
    elif sampling_method and sampling_method != 'none':
        print(f"    Warning: Sampler '{sampling_method}' requested but not implemented in pipeline. No sampling applied.")
    else:
        print(f"    No sampling requested or sampler is 'none'. Using original model.")


    # if sampler: # If a sampler was successfully initialized
    #     # Create the imblearn pipeline
    #     pipeline = ImbPipeline([
    #         ('sampler', sampler),
    #         ('classifier', model_to_tune) # Use the cloned model
    #     ])
    #     # Prefix parameter names in the grid with 'classifier__'
    #     pipeline_param_grid = {f'classifier__{k}': v for k, v in filtered_params.items()}

    #     # Update variables to be used by GridSearchCV
    #     model_to_tune = pipeline
    #     grid_to_use = pipeline_param_grid
    #     print(f"    Using imblearn pipeline for GridSearchCV.")
    
    if sampler: # If a sampler was successfully initialized
        print(f"    Pre-checking sampler '{sampling_method}' effect on full training data:")
        try:
            # Apply sampler to the full training set *just for this check*
            X_train_check, y_train_check = sampler.fit_resample(X_train, y_train)
            print(f"    ---> Counts after applying {sampling_method} (for check): {Counter(y_train_check)}")
            del X_train_check, y_train_check
        except Exception as e:
            print(f"    ---> Could not perform pre-check sampling: {e}")
            # Optionally print traceback 
            # traceback.print_exc()

        # Now, set up the pipeline for GridSearchCV using OG X_train, y_train
        pipeline = ImbPipeline([
            ('sampler', sampler), # The sampler instance is used here
            ('classifier', clone(model)) # Use a clone of the *original* model
        ])
        # Prefix parameter names in the grid with 'classifier__'
        pipeline_param_grid = {f'classifier__{k}': v for k, v in filtered_params.items()}

        # Update variables to be used by GridSearchCV
        model_to_tune = pipeline
        grid_to_use = pipeline_param_grid
        print(f"    Using imblearn pipeline for GridSearchCV (sampling will happen per fold).")
    else:
        # No sampler, use the original cloned model
        model_to_tune = clone(model)
        grid_to_use = filtered_params
        print(f"    Using original model clone for GridSearchCV (no sampling in pipeline).")
    # <<< END: Added block >>>
        
        
    grid_search = GridSearchCV(
        estimator=model_to_tune,      
        param_grid=grid_to_use,       
        cv=stratified_cv,             # stratified_cv
        scoring='f1_weighted',        
        n_jobs=n_jobs,                
        verbose=1,                    
        return_train_score=True,      
        error_score= 'raise',         #  (good for debugging)
    )
    
    
    # # Ensure the grid search does not use test data information
    # grid_search = GridSearchCV(
    #     model, filtered_params, 
    #     # param_grid, 
    #     cv=stratified_cv,  # Use stratified k-fold
    #     scoring='f1_weighted',
    #     n_jobs=n_jobs, 
    #     verbose=1,
    #     return_train_score=True,  # Add this to check for overfitting
    #     error_score= 'raise',  # Return 0 for failed fits instead of raising error
    # )
   
    # Try/except block to catch and debug issues
    try:
        print(f"Starting grid search with CV splits: {getattr(stratified_cv, 'n_splits', 'LOO')}")
        grid_search.fit(X_train, y_train)
    
        # Debug overfitting
        # Print CV results to help diagnose overfitting
        means_train = grid_search.cv_results_['mean_train_score']
        means_test = grid_search.cv_results_['mean_test_score']
        stds_test = grid_search.cv_results_['std_test_score']
        
        print(f"Cross-validation results:")
        print(f"{'Parameters':<40} {'Train Score':<12} {'Test Score':<12} {'Gap':<8}")
        for train_mean, test_mean, std, params in zip(means_train, means_test, stds_test, grid_search.cv_results_['params']):
            # Calculate gap between train and test score (potential overfitting indicator)
            gap = train_mean - test_mean
            param_str = str(params)
            if len(param_str) > 35:
                param_str = param_str[:32] + "..."
            print(f"{param_str:<40} {train_mean:.4f} {test_mean:.4f} (±{std:.4f}) {gap:.4f}")
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        # Check if overfitting is likely (large gap between train and test scores)
        best_idx = grid_search.best_index_
        best_train_score = means_train[best_idx]
        best_test_score = means_test[best_idx]
        gap = best_train_score - best_test_score
        
        params_are_prefixed = bool(sampler) # True if ImbPipeline was used
        current_best_params = grid_search.best_params_
        model_to_regularize = model # The original base model
        
        
        if gap > 0.1:  # More than 10% gap indicates potential overfitting
            print(f"Warning: Possible overfitting (10%+ gap). Train score: {best_train_score:.4f}, Test score: {best_test_score:.4f}, Gap: {gap:.4f}")
            
            # Un-prefix parameters if from a pipeline, before applying regularization to the base model
            params_for_regularization = {}
            if params_are_prefixed:
                print("  Un-prefixing parameters for regularization step...")
                for p_name, p_value in current_best_params.items():
                    if p_name.startswith('classifier__'):
                        params_for_regularization[p_name.split('__', 1)[1]] = p_value
                    else:
                        params_for_regularization[p_name] = p_value # Should not happen if logic is correct
            else:
                params_for_regularization = current_best_params
            
            # Apply regularization if overfitting is severe
            if gap > 0.2:  # More than 20% gap indicates severe overfitting
                print("Severe overfitting detected. Applying stronger regularization...")
                # best_model = apply_regularization(model, grid_search.best_params_, strength=1.5)
                # best_model.fit(X_train, y_train)
                # return best_model, grid_search.best_params_
                best_model_candidate = apply_regularization(model_to_regularize, params_for_regularization, strength=1.5)
                best_model_candidate.fit(X_train, y_train) 
                return best_model_candidate, params_for_regularization
            
            else:
                # Moderate overfitting - apply milder regularization
                print("Moderate overfitting detected. Applying mild regularization...")
                # best_model = apply_regularization(model, grid_search.best_params_, strength=1.2)
                # best_model.fit(X_train, y_train)
                # return best_model, grid_search.best_params_
                best_model_candidate = apply_regularization(model_to_regularize, params_for_regularization, strength=1.2)
                best_model_candidate.fit(X_train, y_train) 
                return best_model_candidate, params_for_regularization
        final_best_estimator = grid_search.best_estimator_
        final_best_params = grid_search.best_params_
        if isinstance(final_best_estimator, ImbPipeline) or isinstance(final_best_estimator, Pipeline):
            print("  Best estimator is a pipeline. Extracting classifier and its original parameters.")
            final_best_estimator = final_best_estimator.named_steps['classifier']
            # Reconstruct unprefixed params that correspond to this classifier
            # This assumes 'filtered_params' holds the original unprefixed grid for the base model
            original_params_for_best_classifier = {}
            for key_prefixed in grid_search.best_params_:
                if key_prefixed.startswith('classifier__'):
                    original_key = key_prefixed.split('__',1)[1]
                    original_params_for_best_classifier[original_key] = grid_search.best_params_[key_prefixed]
            final_best_params = original_params_for_best_classifier
        
        if hasattr(grid_search, 'best_score_') and hasattr(grid_search, 'best_params_'):
            # best_idx = grid_search.best_index_
            # best_train_score = means_train[best_idx] if 'means_train' in locals() else 0.0
            # best_test_score = grid_search.best_score_
            # gap = best_train_score - best_test_score
            
            # Log the results
            log_grid_search_results(
                model_name= model_name, 
                # grid_search.best_params_,
                best_params=final_best_params, 
                train_score=best_train_score, 
                test_score=best_test_score, 
                gap = gap,
                report_dir=report_dir
            )
        
        # return grid_search.best_estimator_, grid_search.best_params_
        return final_best_estimator, final_best_params

        
    except Exception as e:
        print(f"Error during grid search for {model_name}: {str(e)}")
        traceback.print_exc() 
        print("Returning model with conservative parameters")
        # return create_conservative_model(model_name), {}
        conservative_model = create_conservative_model(model_name)
        return conservative_model, conservative_model.get_params() if hasattr(conservative_model, 'get_params') else {}
        

def apply_regularization(model, best_params, strength=1.0):
    """Apply regularization to a model based on its type"""
    from sklearn.base import clone
    
    # Create a clone of the model with best parameters
    regularized_model = clone(model).set_params(**best_params)
    
    # Apply regularization based on model type
    if isinstance(model, RandomForestClassifier):
        # Increase min_samples_leaf and reduce max_depth
        max_depth = regularized_model.max_depth
        if max_depth is not None:
            regularized_model.max_depth = max(3, int(max_depth / strength))
        
        # Increase min_samples_leaf
        min_samples_leaf = regularized_model.min_samples_leaf
        regularized_model.min_samples_leaf = max(2, int(min_samples_leaf * strength))
        
        # Use sqrt feature selection
        regularized_model.max_features = 'sqrt'
        
    elif isinstance(model, GradientBoostingClassifier):
        # Reduce learning rate
        regularized_model.learning_rate = regularized_model.learning_rate / strength
        
        # Reduce max_depth
        max_depth = regularized_model.max_depth
        regularized_model.max_depth = max(2, int(max_depth / strength))
        
        # Increase subsampling
        regularized_model.subsample = max(0.7, min(0.9, regularized_model.subsample * 0.9))
        
    elif isinstance(model, xgb.XGBClassifier):
        # Reduce learning rate
        regularized_model.learning_rate = regularized_model.learning_rate / strength
        
        # Increase regularization parameters
        regularized_model.reg_alpha = regularized_model.reg_alpha * strength
        regularized_model.reg_lambda = regularized_model.reg_lambda * strength
        
        # Reduce max_depth
        max_depth = regularized_model.max_depth
        regularized_model.max_depth = max(2, int(max_depth / strength))
        
        # Increase min_child_weight
        regularized_model.min_child_weight = regularized_model.min_child_weight + 1
        
    elif isinstance(model, LogisticRegression):
        # Decrease C (increase regularization)
        regularized_model.C = regularized_model.C / strength
        
    elif isinstance(model, SVC):
        # Decrease C (increase regularization)
        regularized_model.C = regularized_model.C / strength
        
    return regularized_model
def create_conservative_model(model_name):
    """Create a more conservative model to prevent overfitting"""
    if 'Logistic Regression' in model_name:
        return LogisticRegression(
            max_iter=3000, 
            C=0.1,  # Stronger regularization
            class_weight='balanced',
            random_state=42,
            solver='liblinear'  # Works better for small datasets
        )
    elif 'SVM' in model_name:
        return SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    elif 'Random Forest' in model_name:
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,  # Limit tree depth
            min_samples_leaf=5,  # Require more samples per leaf
            class_weight='balanced',
            random_state=42
        )
    elif 'Gradient Boosting' in model_name:
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.01,  # Slower learning rate
            max_depth=3,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42
        )
    elif 'XGBoost' in model_name:
        return xgb.XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.01,
            max_depth=3,
            reg_alpha=1.0,
            reg_lambda=1.0,
            subsample=0.8,
            random_state=42
        )
    else:
        # Default fallback
        return RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        )
def create_base_models(num_classes, class_distribution=None):
    """
    Create a dictionary of base models with configurations optimized for 
    small imbalanced datasets.
    
    Args:
        num_classes: Number of classes (2 for binary, >2 for multiclass)
        class_distribution: Optional distribution of classes (for calculating class weights)
    
    Returns:
        Dictionary of base models
    """
    is_binary = (num_classes == 2)
    
    # Calculate class weights if distribution is provided
    if class_distribution is not None:
        # For binary: higher weight to minority class
        if is_binary:
            ratio = class_distribution[0] / max(class_distribution[1], 1)
            binary_weight = {0: 1, 1: ratio} if class_distribution[0] > class_distribution[1] else {0: ratio, 1: 1}
        else:
            # Auto-balanced weights are fine for multiclass
            binary_weight = None
    else:
        binary_weight = None
    
    base_models_1 = {
        # 'Logistic Regression': LogisticRegression(
        #     max_iter=5000,  # Increase iterations for convergence
        #     C=0.1,          # Start with stronger regularization
        #     class_weight='balanced' if binary_weight is None else binary_weight,
        #     random_state=42,
        #     multi_class='auto',
        #     solver='liblinear' if is_binary else 'saga'  # liblinear is better for binary
        # )
        
        'Random Forest': RandomForestClassifier(
            n_estimators=300,   # More trees for stability
            max_depth=5,        # Limit depth to prevent overfitting
            # min_samples_leaf=5, # Require more samples per leaf
            min_samples_leaf = 2,
            min_samples_split = 5,
            class_weight= "balanced_subsample", #'balanced',
            max_features='sqrt',  # Feature subsampling
            criterion = "entropy",
            random_state=42
        )
    }
    
    base_models = {
        'Logistic Regression': LogisticRegression(
            max_iter=5000,  # Increase iterations for convergence
            C=0.1,          # Start with stronger regularization
            class_weight='balanced' if binary_weight is None else binary_weight,
            random_state=42,
            multi_class='auto',
            solver='liblinear' if is_binary else 'saga'  # liblinear is better for binary
        ),
        'SVM': SVC(
            kernel='rbf',    # rbf usually works well 
            C=1.0,           # Moderate regularization
            gamma='scale',   # Adaptive gamma based on data
            degree=2,        # For polynomial kernel
            class_weight='balanced',
            decision_function_shape = "ovr" if not is_binary else "ovo",  # one-vs-rest for multiclass, one-vs-one otherwise
            probability=True,
            random_state=42
        ),
        'SVM_pipeline': Pipeline([
            ('scale', StandardScaler()),  # Always scale for SVM
            ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ('svm', SVC(
                kernel='rbf', 
                C=1.0, 
                gamma='scale', 
                class_weight='balanced',
                probability=True, 
                random_state=42
            ))
        ]),
        'Random Forest': RandomForestClassifier(
            n_estimators=300,   # More trees for stability
            max_depth=5,        # Limit depth to prevent overfitting
            # min_samples_leaf=5, # Require more samples per leaf
            min_samples_leaf = 2,
            min_samples_split = 5,
            class_weight= "balanced_subsample", #'balanced',
            max_features='sqrt',  # Feature subsampling
            criterion = "entropy",
            random_state=42
        ),
        'RF_pipleline' : Pipeline([
                ('feature_selection', SelectKBest(score_func=f_classif)),
                ('classification', RandomForestClassifier(random_state=42, class_weight='balanced')) # Use balanced weights
            ]),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.01,      # Slower learning rate
            max_depth=3,             # Shallow trees
            subsample=0.8,           # Row subsampling
            min_samples_split=10,    # Require more samples to split
            min_samples_leaf=5,      # Require more samples per leaf
            validation_fraction=0.2, # Use 20% for early stopping validation
            n_iter_no_change=20,     # Early stopping patience
            tol=0.001,               # Early stopping tolerance
            random_state=42
        ),
        'XGBoost': xgb.XGBClassifier(
            objective="binary:logistic" if is_binary else "multi:softprob", #"multi:softmax",
            num_class=None if is_binary else 4,
            eval_metric='logloss',
            max_depth=2,               # Reduce from 3 to 2
            learning_rate=0.005,       # Reduce from 0.01
            n_estimators=200,          # Reduce from 200
            subsample=0.7,             # Reduce from 0.8
            colsample_bytree=0.6,      # Reduce from 0.8
            reg_alpha=2.0,             # Increase from 0.5
            reg_lambda=3.0,            # Increase from 1.0
            min_child_weight=5,
            random_state = 42 # Increase from 3
            # scale_pos_weight=1.5 if is_binary else 1
        ),
        # Inside the base_models dictionary
        'Gaussian NB': GaussianNB(),
        # add others here AFTER preprocess data for them
        # 'Complement NB': ComplementNB(), # Requires non-negative features
        # 'Bernoulli NB': BernoulliNB(),  # Requires binary [0,1] features
        }
    
    return base_models # 
    # return base_models_all

def filter_param_grid(param_grid):
    """Filter parameter grid to remove incompatible combinations"""
    from sklearn.model_selection import ParameterGrid
    
    # Convert ParameterGrid back to a dictionary format that GridSearchCV expects
    valid_params = {}
    for key in param_grid:
        valid_params[key] = param_grid[key]  # Keep original format
    
    # Filter out incompatible combinations from elasticnet penalty
    if 'penalty' in param_grid and 'l1_ratio' in param_grid:
        # If penalty is not elasticnet, don't include l1_ratio
        if 'elasticnet' not in param_grid['penalty']:
            valid_params.pop('l1_ratio', None)
    
    print(f"Filtered parameter grid: {valid_params}")
    return param_grid

# Define specialized parameter grids for binary classification
binary_param_grids = {
    'Logistic Regression': {
        'C': [0.0001, 0.001, 0.01, 0.1],  # Focus on smaller C (stronger regularization)
        'penalty': ['l2'],
        'solver': ['saga'], # Needed for elasticnet if you re-add it, works for l2
        'class_weight': ['balanced']
    },
    'SVM': {
        'C': [0.01, 0.1, 1.0], # Smaller C range
        'kernel': ['linear', 'rbf'],  # Prioritize simpler kernels
        'gamma': ['scale', 'auto'], # Avoid large gamma values for rbf
        # 'degree': [2], # Keep poly simple if re-added, but maybe remove for now
        'class_weight': ['balanced', {0: 1, 1: 2}] # Reduce weight complexity
    },
    # Consider simplifying or removing the SVM pipeline for initial overfitting checks
    'SVM_pipeline': {
        'poly__degree': [1], # Only linear features initially
        'svm__C': [0.1, 1.0],
        'svm__gamma': ['scale'],
        'svm__kernel': ['rbf'],
        'svm__class_weight': ['balanced']
    },
    'Random Forest': {
        'n_estimators': [50, 100], # Fewer trees
        'max_depth': [2, 3, 4], # Much shallower trees
        'min_samples_split': [10, 20], # Increase minimum split size
        'min_samples_leaf': [5, 10], # Increase minimum leaf size
        'max_features': ['sqrt', 0.5], # Keep feature sampling restricted
        'class_weight': ['balanced', 'balanced_subsample']
    },
    'RF_pipleline': {
        'feature_selection__k': [10, 20, 40], # Tune number of features
        'classification__n_estimators': [50, 100], # Note the prefix 'classification__'
        'classification__max_depth': [3, 4], # Shallower still
        'classification__min_samples_split': [15, 25], # More regularization
        'classification__min_samples_leaf': [10, 15], # More regularization
        'classification__max_features': ['sqrt']
        # Class weight is now fixed in the pipeline definition above
    },
    
    'Gradient Boosting': {
        'n_estimators': [50, 100], # Fewer trees
        'learning_rate': [0.01, 0.05, 0.1], # Slightly faster rates ok with fewer trees
        'max_depth': [2, 3], # Shallow trees only
        'subsample': [0.6, 0.7], # Stronger subsampling
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10],
        'max_features': ['sqrt', 0.5] # Restrict features per split
    },
    'XGBoost': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.05],
        'max_depth': [1, 2, 3], # Include stumps (depth 1)
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'reg_alpha': [1.0, 10.0, 50.0], # Stronger L1
        'reg_lambda': [1.0, 10.0, 50.0], # Stronger L2
        'min_child_weight': [3, 5, 10], # Higher min child weight
        'scale_pos_weight': [1, 3] # Keep balanced options
    },
    'Gaussian NB': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    },
}

# Multiclass grids are slightly different
multiclass_param_grids = {
    'Logistic Regression': {
        'C': [0.001, 0.01, 0.1],
        'penalty': ['l2'], # multinomial often prefers l2
        'solver': ['saga'],
        'multi_class': ['multinomial'],
        'class_weight': ['balanced']
    },
    'SVM': {
        'C': [0.1, 1.0, 5.0], # Slightly higher C might be needed for multiclass separation
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale'],
        'decision_function_shape': ['ovr'], # One-vs-Rest is often default
        'class_weight': ['balanced']
    },
    'SVM_pipeline': {
         'poly__degree': [1],
         'svm__C': [0.1, 1.0],
         'svm__gamma': ['scale'],
         'svm__kernel': ['rbf'],
         'svm__decision_function_shape': ['ovr'],
         'svm__class_weight': ['balanced']
     },
    'Random Forest': {
        'n_estimators': [50, 100, 300],
        'max_depth': [3, 4, 5], # Shallow
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10],
        'max_features': ['sqrt'],
        'class_weight': ['balanced', 'balanced_subsample']
    },
    'RF_pipleline': {
        'feature_selection__k': [40, 80, 120], # Tune number of features
        'classification__n_estimators': [50, 100], # Note the prefix 'classification__'
        'classification__max_depth': [3, 4], # Shallower still
        'classification__min_samples_split': [15, 25], # More regularization
        'classification__min_samples_leaf': [10, 15], # More regularization
        'classification__max_features': ['sqrt']
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.05],
        'max_depth': [2, 3],
        'subsample': [0.6, 0.7],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10],
        'max_features': ['sqrt']
    },
    'XGBoost': {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.05],
        'max_depth': [2, 3],
        'subsample': [0.5, 0.7],
        'colsample_bytree': [0.5, 0.7],
        'reg_alpha': [1.0, 10.0],
        'reg_lambda': [1.0, 10.0],
        'min_child_weight': [3, 5, 10]
        # 'objective': 'multi:softmax' # Will be set based on num_classes
        # 'num_class': # Will be set based on num_classes
    },
    'Gaussian NB': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    },
}
    
#------------------------End--------------------------------


def plot_learning_curve(model, X_train, y_train, X_test, y_test, cv=3, n_jobs=1):
    """
    Plot learning curves to diagnose bias/variance problems with automatic fold adjustment
    
    Args:
        model: Model to evaluate
        X_train, y_train: Training data
        X_test, y_test: Test data
        cv: Maximum number of cross-validation folds
        n_jobs: Number of jobs for parallel processing
    """
    from sklearn.model_selection import learning_curve
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from sklearn.metrics import f1_score
    
    # Check if cv is too large for the minority class
    class_counts = np.bincount(y_train)
    min_samples = min(class_counts[class_counts > 0])  # Ignore classes with zero samples
    
    # Adjust cv if necessary
    original_cv = cv
    if min_samples < cv:
        # Use at most 3 splits, but not more than min_samples
        cv = max(2, min(3, min_samples))
        print(f"\nAdjusting learning curve: Minority class has only {min_samples} samples, "
              f"reducing folds from {original_cv} to {cv}")
    
    # Limit n_jobs to avoid memory issues
    # n_jobs = min(2, os.cpu_count() or 1)  # Use at most 2 processes
    n_jobs =  1
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    try:
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
        plt.figure(figsize=(10, 6))
        plt.title('Learning Curves')
        plt.xlabel('Training Examples')
        plt.ylabel('F1 Score')
        plt.grid()
        
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
        plt.fill_between(train_sizes, validation_mean - validation_std, validation_mean + validation_std, alpha=0.1, color="green")
        plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training score")
        plt.plot(train_sizes, validation_mean, 's-', color="green", label="Cross-validation score")
        plt.close()
        
        # Plot test score if available
        if X_test is not None and y_test is not None:
            test_score = f1_score(y_test, model.predict(X_test), average='weighted')
            plt.axhline(y=test_score, color='r', linestyle='-', label=f'Test score: {test_score:.4f}')
        
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
        plt.close()
        
        return train_sizes, train_scores, validation_scores
    
    except Exception as e:
        print(f"\nWarning: Could not generate learning curve: {e}")
        plt.figure(figsize=(10, 6))
        plt.title('Learning Curve Generation Failed')
        plt.xlabel('Error occurred during computation')
        plt.ylabel('F1 Score')
        plt.text(0.5, 0.5, f"Error: {str(e)}", horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.close()
        
        return None, None, None

def get_standardized_model_name(model):
    """Get a standardized name for a model that's consistent across plots"""
    # Start with the class name
    model_name = type(model).__name__
    
    # Special cases for common models
    if model_name == 'RandomForestClassifier':
        return 'Random Forest'
    elif model_name == 'GradientBoostingClassifier':
        return 'Gradient Boosting'
    elif model_name == 'LogisticRegression':
        return 'Logistic Regression'
    elif model_name == 'XGBClassifier':
        return 'XGBoost'
    elif model_name == 'SVC':
        return 'SVM'
    elif model_name == 'Pipeline':
        # For pipelines, use the last step name
        if hasattr(model, 'steps') and len(model.steps) > 0:
            last_step_name = model.steps[-1][0]
            if last_step_name == 'svm':
                return 'SVM_pipeline'
            return last_step_name
    elif "Stacking" in model_name:
        return "Stacking Classifier"
    elif "Voting" in model_name:
        return "Voting Classifier"
    return model_name




# works!! global shap
def plot_circular_shap_heatmap(shap_values, feature_names, num_zones_per_metric, model_name_prefix, shap_output_dir):
    """
    Generates a composite circular SHAP heatmap for all feature groups by plotting
    annulus segments on a Cartesian axis to simulate a polar view.

    Args:
        shap_values (np.ndarray): 2D array of SHAP values (samples, features).
        feature_names (list): List of the base feature group names (e.g., ['std', 'skew']).
        num_zones_per_metric (int): The number of zones per feature group (e.g., 20).
        model_name_prefix (str): A name for the model to use in titles and filenames.
        shap_output_dir (Path or str): The directory where the plot image will be saved.
    """
    logger.info(f"Generating composite circular SHAP heatmaps for {model_name_prefix}")
    
    # Ensure output directory exists
    shap_output_dir = Path(shap_output_dir)
    shap_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # --- Step 1: Prepare the Data ---
        mean_abs_shaps_flat = np.abs(shap_values).mean(axis=0)
        num_metrics = len(feature_names)

        expected_total_features = num_metrics * num_zones_per_metric
        if mean_abs_shaps_flat.shape[0] != expected_total_features:
            logger.error(f"SHAP values dimension mismatch ({mean_abs_shaps_flat.shape[0]}) vs expected ({expected_total_features}). Cannot create heatmap.")
            return

        mean_abs_shaps_reshaped = mean_abs_shaps_flat.reshape(num_metrics, num_zones_per_metric)
        heatmap_data = pd.DataFrame(mean_abs_shaps_reshaped, index=feature_names, columns=[f"Z{i+1}" for i in range(num_zones_per_metric)])
        
        if heatmap_data.empty:
            logger.info("Skipping circular SHAP plot as heatmap_data is empty.")
            return

        # --- Step 2: Define the Circular Plot Geometry ---
        r_inner, r_middle, r_outer = 0.3, 0.6, 1.0
        zone_definitions = []
        start_segment_offsets = {"inner": 0, "middle": 1, "outer": 1}

        # Inner Ring (4 segments, Z1-Z4)
        angles_inner_vad = np.linspace(0, 360, 4 + 1)
        for i in range(4):
            k = (start_segment_offsets["inner"] - i) % 4
            label = 1 + k
            zone_definitions.append({"data_col_name": f"Z{label}", "display_label": str(label), "r": r_inner, "theta1": angles_inner_vad[i], "theta2": angles_inner_vad[i+1], "annulus_width": r_inner})

        # Middle Ring (8 segments, Z5-Z12)
        angles_middle_vad = np.linspace(0, 360, 8 + 1)
        for i in range(8):
            k = (start_segment_offsets["middle"] - i) % 8
            label = 5 + k
            zone_definitions.append({"data_col_name": f"Z{label}", "display_label": str(label), "r": r_middle, "theta1": angles_middle_vad[i], "theta2": angles_middle_vad[i+1], "annulus_width": r_middle - r_inner})

        # Outer Ring (8 segments, Z13-Z20)
        angles_outer_vad = np.linspace(0, 360, 8 + 1)
        for i in range(8):
            k = (start_segment_offsets["outer"] - i) % 8
            label = 13 + k
            zone_definitions.append({"data_col_name": f"Z{label}", "display_label": str(label), "r": r_outer, "theta1": angles_outer_vad[i], "theta2": angles_outer_vad[i+1], "annulus_width": r_outer - r_middle})

        # --- Step 3: Create the Composite Plot ---
        ncols = 3
        nrows = int(np.ceil(num_metrics / ncols))
        master_fig, master_axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows + 0.7))
        master_axes_flat = master_axes.flatten() if num_metrics > 1 else [master_axes]

        g_vmin, g_vmax = heatmap_data.min().min(), heatmap_data.max().max()
        if g_vmin == g_vmax: g_vmin -= 0.001; g_vmax += 0.001
        norm = plt.Normalize(vmin=g_vmin, vmax=g_vmax)
        cmap = plt.cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])

        for feature_idx, feature_name in enumerate(heatmap_data.index):
            if feature_idx >= len(master_axes_flat): break
            ax = master_axes_flat[feature_idx]
            ax.set_aspect('equal', adjustable='box')
            current_feature_shap_values = heatmap_data.loc[feature_name]

            for zone_def in zone_definitions:
                if zone_def["data_col_name"] in current_feature_shap_values:
                    shap_value = current_feature_shap_values[zone_def["data_col_name"]]
                    color = cmap(norm(shap_value))

                    r_outer_segment = zone_def["r"]
                    r_inner_segment = zone_def["r"] - zone_def["annulus_width"]
                    
                    wedge_theta1_deg = (90 - zone_def["theta2"] + 360) % 360
                    wedge_theta2_deg = (90 - zone_def["theta1"] + 360) % 360
                    
                    wedge = patches.Wedge(
                        center=(0, 0), r=r_outer_segment,
                        theta1=wedge_theta1_deg, theta2=wedge_theta2_deg,
                        width=r_outer_segment - r_inner_segment,
                        facecolor=color, edgecolor='black', linewidth=0.3
                    )
                    ax.add_patch(wedge)

                    # ADD ZONE NUMBER LABELS ***
                    # Calculate the center of the wedge to place the label
                    label_vad_center = (zone_def["theta1"] + zone_def["theta2"]) / 2
                    label_radius = r_outer_segment - (zone_def["annulus_width"] / 2)
                    if r_inner_segment == 0:  # Adjust radius for the inner-most circle
                        label_radius = r_outer_segment / 1.6

                    # Convert polar position to Cartesian for the text label
                    label_mad_rad = np.deg2rad((90 - label_vad_center + 360) % 360)
                    x_label = label_radius * np.cos(label_mad_rad)
                    y_label = label_radius * np.sin(label_mad_rad)

                    # Add the text with contrasting color for visibility
                    ax.text(x_label, y_label, zone_def["display_label"],
                            ha='center', va='center', fontsize=10, # Small font for clarity
                            color="white" if norm(shap_value) < 0.3 else "black")

            ax.set_xlim(-r_outer - 0.05, r_outer + 0.05)
            ax.set_ylim(-r_outer - 0.05, r_outer + 0.05)
            ax.axis('off')
            ax.set_title(feature_name, fontsize=18)

        for i in range(num_metrics, len(master_axes_flat)):
            master_axes_flat[i].axis('off')

        cbar_ax = master_fig.add_axes([0.92, 0.15, 0.015, 0.7])
        # master_fig.colorbar(sm, cax=cbar_ax, label="Mean(|SHAP Value|)")
        #increase foontsize
        cbar = master_fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("Mean(|SHAP Value|)", fontsize=20)
        cbar.ax.tick_params(labelsize=15)

        master_fig.suptitle(f"Circular SHAP Heatmaps for {model_name_prefix}", fontsize=20, y=0.98)
        master_fig.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        save_path = shap_output_dir / f"{model_name_prefix}_shap_circular_heatmap_composite_glb.png"
        master_fig.savefig(save_path, dpi=150)
        plt.close(master_fig)
        logger.info(f"Saved composite circular SHAP heatmap to {save_path}")

    except Exception as e:
        logger.error(f"Error in plot_circular_shap_heatmap: {e}", exc_info=True)


def plot_circular_shap_heatmap_final(shap_values_array, group_metrics, num_zones_per_metric, model_name_prefix, shap_output_dir, plot_title, filename_suffix, is_global):
    """
    A robust, single function to generate a circular SHAP heatmap for both global and instance plots.
    """
    logger.info(f"Generating circular SHAP heatmap: {plot_title}")
    
    total_shap_features = shap_values_array.shape[-1]
    expected_features = len(group_metrics) * num_zones_per_metric
    
    if total_shap_features != expected_features:
        logger.error(f"Cannot create circular heatmap for '{plot_title}'. Feature count mismatch. Expected {expected_features}, but got {total_shap_features}.")
        return
        
    num_metrics = len(group_metrics)

    try:
        heatmap_data = pd.DataFrame(shap_values_array.reshape(num_metrics, num_zones_per_metric), index=group_metrics)
    except ValueError as e:
        logger.error(f"Failed to reshape SHAP values for heatmap '{plot_title}': {e}")
        return

    r_inner, r_middle, r_outer = 0.3, 0.6, 1.0
    zone_definitions = []
    
    angles_inner = np.linspace(0, 360, 4 + 1)
    for i in range(4): zone_definitions.append({"data_col_idx": i, "display_label": str(i + 1), "r": r_inner, "theta1": angles_inner[i], "theta2": angles_inner[i+1], "annulus_width": r_inner})
    
    angles_middle = np.linspace(0, 360, 8 + 1)
    for i in range(8): zone_definitions.append({"data_col_idx": i + 4, "display_label": str(i + 5), "r": r_middle, "theta1": angles_middle[i], "theta2": angles_middle[i+1], "annulus_width": r_middle-r_inner})

    angles_outer = np.linspace(0, 360, 8 + 1)
    for i in range(8): zone_definitions.append({"data_col_idx": i + 12, "display_label": str(i + 13), "r": r_outer, "theta1": angles_outer[i], "theta2": angles_outer[i+1], "annulus_width": r_outer-r_middle})

    ncols = 3
    nrows = int(np.ceil(num_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows + 0.7))
    master_axes_flat = axes.flatten() if num_metrics > 1 else [axes]

    if is_global:
        cmap, norm = plt.cm.viridis, colors.Normalize(vmin=heatmap_data.values.min(), vmax=heatmap_data.values.max())
        cbar_label = "Mean Abs SHAP Value"
    else:
        vmax = np.abs(heatmap_data.values).max()
        cmap, norm = plt.cm.RdBu, colors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
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


# eval run shap()
def run_shap_analysis_eval(model, X_test, feature_names, report_dir, config, y_test=None, sample_indices_to_plot=[9]):
    """
    Definitive, robust function for SHAP analysis. This version uses a
    resilient hybrid approach to handle inconsistencies in SHAP plotting functions.
    """
    model_name_str = get_standardized_model_name(model)
    logger.info(f"--- Running Definitive SHAP Analysis for {model_name_str} ---")

    try:
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        base_model = model.steps[-1][1] if isinstance(model, Pipeline) else model
        figures_dir = Path(report_dir) / 'figures' if report_dir else None
        if figures_dir: figures_dir.mkdir(parents=True, exist_ok=True)

        # 1. Robust Explainer Initialization
        logger.info(f"  Initializing SHAP explainer for {type(base_model).__name__}...")
        try:
            if isinstance(base_model, (RandomForestClassifier, xgb.XGBClassifier, LogisticRegression)):
                explainer = shap.Explainer(base_model, X_test_df)
            elif isinstance(base_model, GradientBoostingClassifier):
                masker = shap.maskers.Independent(X_test_df)
                explainer = shap.TreeExplainer(model, masker)
            else:
                logger.info("    Using KernelExplainer as fallback.")
                background_data = shap.sample(X_test_df, min(100, X_test_df.shape[0]))
                explainer = shap.KernelExplainer(base_model.predict_proba, background_data)
        except Exception as e:
            logger.error(f"  Failed to initialize SHAP explainer: {e}")
            return

        # 2. Calculate SHAP values
        raw_shap_values = explainer.shap_values(X_test_df)
        
        # 3. Handle different explainer output formats
        is_binary = not isinstance(raw_shap_values, list)
        class_indices = [0, 1] if is_binary else range(len(raw_shap_values))

        for class_idx in class_indices:
            class_name = f"Class_{class_idx}"
            logger.info(f"--- Generating plots for Class: {class_name} ---")

            # 4. Extract values for the current class
            shap_values_for_class = raw_shap_values if (is_binary and class_idx == 1) else -raw_shap_values if is_binary else raw_shap_values[class_idx]

            # 5. Generate Plots
            # --- Summary Plot (most stable method) ---
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values_for_class, X_test_df, show=False)
            if figures_dir: plt.savefig(figures_dir / f"{model_name_str}_shap_summary_{class_name}.png", dpi=150, bbox_inches='tight')
            plt.close()

            # For plots that require an Explanation object, create one.
            explanation = shap.Explanation(
                values=shap_values_for_class,
                base_values=explainer.expected_value[class_idx] if not is_binary else explainer.expected_value,
                data=X_test_df,
                feature_names=feature_names
            )

            # --- Bar Plot (only for positive class) ---
            if not is_binary or class_idx == 1:
                plt.figure(figsize=(10, 8))
                shap.plots.bar(explanation, show=False)
                if figures_dir: plt.savefig(figures_dir / f"{model_name_str}_shap_bar_{class_name}.png", dpi=150, bbox_inches='tight')
                plt.close()
            
            # --- Waterfall & Grouped Waterfall ---
            all_metrics = ['mean', 'median', 'std', 'iqr', 'idr', 'skew', 'kurt', 'Del', 'Amp']
            feature_indices = config.get('feature_indices')
            group_metrics = [all_metrics[i] for i in feature_indices] if feature_indices is not None else all_metrics
            values_per_group = config.get('values_per_feature', 20)
            
            for sample_idx in sample_indices_to_plot:
                if sample_idx >= X_test_df.shape[0]: continue
                
                
                # Standard Waterfall
                instance_explanation = explanation[sample_idx]

                # Calculate probability from the logit 
                final_logit = instance_explanation.base_values + instance_explanation.values.sum()
                probability = 1 / (1 + np.exp(-final_logit))

                # Create the title with percentage certainty
                logger.info(f"SHAP Waterfall (Inst {sample_idx}, Class {class_name}) - {model_name_str}\n"
                  f"Certainty: {probability:.2%} | Logit: {final_logit:.2f}")
                # waterfall_title = (f"SHAP Waterfall ({class_name}) - Inst: {sample_idx} | Certainty: {probability:.2%} (Logit: {final_logit:.2f}\n"
                #                 f"{model_name_str}")

                
                plt.figure(figsize=(10, 8))
                shap.plots.waterfall(explanation[sample_idx], show=False)
                plt.title(f"SHAP Waterfall (Inst {sample_idx}, {class_name}) - Probability:{probability:.2%}\n"
                    f"{model_name_str}")
                if figures_dir: plt.savefig(figures_dir / f"{model_name_str}_shap_waterfall_{class_name}_sample{sample_idx}_P{probability:.2%}.png", dpi=150, bbox_inches='tight')
                plt.close()

                # Grouped Waterfall
                grouped_vals = [explanation[sample_idx, i*values_per_group:(i+1)*values_per_group].values.sum() for i in range(len(group_metrics))]
                grouped_expl = shap.Explanation(values=np.array(grouped_vals), base_values=explanation[sample_idx].base_values, feature_names=group_metrics)
                
                plt.figure(figsize=(10, 8))
                shap.plots.waterfall(grouped_expl, show=False)
                if figures_dir: plt.savefig(figures_dir / f"{model_name_str}_shap_grouped_waterfall_{class_name}_sample{sample_idx}.png", dpi=150, bbox_inches='tight')
                plt.close()

            # 6. Generate Circular Plots
            if not is_binary or class_idx == 1:
                plot_circular_shap_heatmap_final(
                    shap_values_array=np.abs(shap_values_for_class).mean(axis=0), group_metrics=group_metrics,
                    num_zones_per_metric=values_per_group, model_name_prefix=model_name_str, shap_output_dir=figures_dir,
                    plot_title=f"Global Mean(|SHAP|) - {model_name_str} ({class_name})", filename_suffix=f"global_{class_name}", is_global=True
                )
            for sample_idx in sample_indices_to_plot:
                if sample_idx < X_test_df.shape[0]:
                    plot_circular_shap_heatmap_final(
                        shap_values_array=shap_values_for_class[sample_idx], group_metrics=group_metrics,
                        num_zones_per_metric=values_per_group, model_name_prefix=model_name_str, shap_output_dir=figures_dir,
                        plot_title=f"Instance SHAP - {model_name_str} (Sample {sample_idx}, {class_name})", filename_suffix=f"instance_{sample_idx}_{class_name}", is_global=False
                    )
        # 7. Generate Feature Group Importance Plot
        logger.info(f"  Generating global feature group importance plot... ") #{class_name}...")
        try:
            analyze_feature_group_importance(
                model=model,
                X_test=X_test,
                num_features=len(group_metrics),
                values_per_feature=values_per_group,
                metrics=group_metrics,
                class_index=1, #class_idx,
                report_dir=figures_dir, 
                y_test=y_test
            )
        except Exception as e:
            logger.error(f"  Failed to generate feature group importance plot for {class_name}: {e}", exc_info=True)        
            
    except Exception as e:
        logger.error(f"Definitive SHAP analysis failed for {model_name_str}: {e}", exc_info=True)



class ShapTimeoutError(Exception):
    """Custom exception for SHAP calculation timeout."""
    pass

# --- Signal Handler Function ---
def shap_timeout_handler(signum, frame):
    """Raises ShapTimeoutError when signal is received."""
    print("\n!!! SHAP CALCULATION TIMEOUT !!!")
    raise ShapTimeoutError("SHAP calculation exceeded the allowed time.")

   


def run_shap_analysis(model, X_test, feature_names=None, sample_idx=None,
                          plot_type='summary', class_index=0, max_display=20,
                          report_dir=None,
                          background_samples=100, explain_samples=100,
                          num_feature_groups=9, values_per_group=20,
                          group_metrics=None,
                          is_binary=None, # Will be determined if None
                          y_test=None): # y_test is optional, used by some fallbacks
    """
    Run SHAP analysis on a model and save the visualizations.
    Includes grouped waterfall and robust error handling.
    """
    try:
        # --- Determine Binary/Multiclass Status ---
        if is_binary is None:
            if hasattr(model, 'n_classes_') and model.n_classes_ == 2:
                is_binary = True
            elif hasattr(model, 'classes_') and len(model.classes_) == 2:
                is_binary = True
            elif y_test is not None and len(np.unique(y_test)) == 2:
                is_binary = True
            else: # Fallback or if multiclass
                is_binary = False
        print(f"\nRunning SHAP analysis (plot_type='{plot_type}', is_binary={is_binary}) for {get_standardized_model_name(model)}")

        # --- Model Handling: Get Base Model if Pipeline ---
        base_model = model.steps[-1][1] if isinstance(model, Pipeline) else model
        is_stacking = isinstance(base_model, StackingClassifier) # Check if base_model is Stacking

        # --- Feature Names Handling ---
        num_original_features = X_test.shape[1]
        if feature_names is None or len(feature_names) != num_original_features:
            print(f"  Warning: Feature names issue (Given: {len(feature_names) if feature_names else 'None'}, Expected: {num_original_features}). Using generic names.")
            feature_names = [f'feature_{i}' for i in range(num_original_features)]
        current_feature_names = feature_names

        # --- Data Preparation for SHAP ---
        # Ensure X_test is a DataFrame for SHAP Explanation object consistency
        if not isinstance(X_test, pd.DataFrame):
            X_test_df = pd.DataFrame(X_test, columns=current_feature_names)
        else:
            X_test_df = X_test.copy() # Use a copy to avoid modifying original

        # --- Initialize Variables for Saving ---
        model_name_str = get_standardized_model_name(model)
        file_prefix = f"{model_name_str}_shap"
        filename = ""
        save_path = None
        figures_dir = None

        if report_dir:
            try:
                figures_dir = Path(report_dir) / 'figures'
                figures_dir.mkdir(parents=True, exist_ok=True)
            except Exception as dir_e:
                print(f"  ERROR: Could not create figures directory '{figures_dir}': {dir_e}. Plot saving will be skipped.")
                figures_dir = None # Disable saving

        # --- SHAP Value Calculation ---
        shap_values = None
        explainer_obj = None # To store the explainer instance
        X_explain = X_test_df # Use DataFrame for explanation
        
        # Reduce X_explain for KernelExplainer if too large to prevent memory/time issues
        if not isinstance(base_model, (RandomForestClassifier, xgb.XGBClassifier, #GradientBoostingClassifier,
                                       LogisticRegression, LinearRegression, Ridge, Lasso)) \
            and not (isinstance(base_model, SVC) and base_model.kernel == 'linear')\
            and not isinstance(base_model, GradientBoostingClassifier): # Exclude GBT:
                if X_explain.shape[0] > explain_samples:
                    print(f"  Subsampling X_test for KernelExplainer from {X_explain.shape[0]} to {explain_samples} samples.")
                    X_explain = shap.sample(X_explain, explain_samples, random_state=42)


        try:
            print(f"  Attempting to select SHAP explainer for: {type(base_model).__name__}")
            
            # Determine if the task is multiclass for GBT handling
            is_multiclass_task_for_gbt = not is_binary # is_binary is passed to run_shap_analysis

            if isinstance(base_model, GradientBoostingClassifier) and is_multiclass_task_for_gbt:
                print(f"    GradientBoostingClassifier is multiclass. Using KernelExplainer due to SHAP limitations.")
                # Fallback to KernelExplainer directly for multiclass GBT
                if X_explain.shape[0] > explain_samples: # Subsample specifically for GBT KernelExplainer
                    print(f"  Subsampling X_test for GBT KernelExplainer from {X_explain.shape[0]} to {explain_samples} samples.")
                    X_explain_gbt = shap.sample(X_explain, explain_samples, random_state=42)
                else:
                    X_explain_gbt = X_explain
                
                background_data_df = shap.sample(X_test_df, min(background_samples, X_test_df.shape[0]), random_state=42) #
                predict_fn_for_kernel = base_model.predict_proba if hasattr(base_model, 'predict_proba') else base_model.predict #
                explainer_obj = shap.KernelExplainer(predict_fn_for_kernel, background_data_df) #
                print(f"      Explaining {X_explain_gbt.shape[0]} samples with KernelExplainer for GBT...") #
                shap_values = explainer_obj.shap_values(X_explain_gbt, nsamples='auto', l1_reg='aic') #

            elif isinstance(base_model, (RandomForestClassifier, GradientBoostingClassifier )): #xgb.XGBClassifier, OG
                print(f"    Using TreeExplainer for {type(base_model).__name__}")
                explainer_obj = shap.TreeExplainer(base_model, data=X_explain, model_output="raw", feature_perturbation="interventional") #if type(base_model).__name__ != 'XGBClassifier' else 'tree_path_dependent')  #Using separate flow with interventional for xgb
                shap_values = explainer_obj.shap_values(X_explain, check_additivity=False)
            elif isinstance(base_model, xgb.XGBClassifier):
                print(f"    Using TreeExplainer for XGBClassifier with interventional perturbation.") #
                explainer_obj = shap.TreeExplainer(base_model, data=X_explain, model_output="raw", feature_perturbation="interventional") #
                shap_values = explainer_obj.shap_values(X_explain, check_additivity=False) #
            elif isinstance(base_model, (LogisticRegression, LinearRegression, Ridge, Lasso)) or \
                    (isinstance(base_model, SVC) and base_model.kernel == 'linear'):
                print(f"    Using LinearExplainer for {type(base_model).__name__}")
                explainer_obj = shap.LinearExplainer(base_model, X_explain)
                shap_values = explainer_obj.shap_values(X_explain)
            else: # Fallback to KernelExplainer
                print(f"    Using KernelExplainer as fallback for {type(base_model).__name__}")
                # Ensure background data is DataFrame with correct column names
                background_data_df = shap.sample(X_test_df, min(background_samples, X_test_df.shape[0]), random_state=42)
                
                predict_fn_for_kernel = base_model.predict_proba if hasattr(base_model, 'predict_proba') else base_model.predict
                if is_stacking: # Custom wrapper for StackingClassifier
                    def wrapped_predict_proba_stacking(X_input_np):
                        # KernelExplainer might pass numpy, convert to DataFrame
                        X_input_df = pd.DataFrame(X_input_np, columns=X_test_df.columns)
                        return base_model.predict_proba(X_input_df)
                    predict_fn_for_kernel = wrapped_predict_proba_stacking

                explainer_obj = shap.KernelExplainer(predict_fn_for_kernel, background_data_df)
                print(f"      Explaining {X_explain.shape[0]} samples with KernelExplainer...")
                shap_values = explainer_obj.shap_values(X_explain, nsamples='auto', l1_reg='aic') # Use 'auto' or a fixed number for nsamples

        except Exception as e_shap_calc:
            # Check if the error is the specific XGBoost background data error
            if "The background dataset you provided does not cover all the leaves" in str(e_shap_calc) and isinstance(base_model, xgb.XGBClassifier):
                print(f"    XGBoost TreeExplainer with default/tree_path_dependent failed: {e_shap_calc}. Retrying with feature_perturbation='interventional'.")
                try:
                    explainer_obj = shap.TreeExplainer(base_model, data=X_explain, feature_perturbation="interventional")
                    shap_values = explainer_obj.shap_values(X_explain, check_additivity=False) # Add check_additivity=False for interventional
                    print("    Successfully used TreeExplainer with interventional perturbation for XGBoost.")
                except Exception as e_retry:
                    print(f"    Retry with interventional perturbation also failed for XGBoost: {e_retry}")
                    traceback.print_exc()
                    # Fallback to KernelExplainer for XGBoost if interventional also fails
                    print("    Falling back to KernelExplainer for XGBoost.")
                    background_data_df = shap.sample(X_test_df, min(background_samples, X_test_df.shape[0]), random_state=42)
                    predict_fn_for_kernel = base_model.predict_proba if hasattr(base_model, 'predict_proba') else base_model.predict
                    explainer_obj = shap.KernelExplainer(predict_fn_for_kernel, background_data_df)
                    shap_values = explainer_obj.shap_values(X_explain, nsamples='auto', l1_reg='aic')

            else:
                print(f"  ERROR during SHAP value calculation: {e_shap_calc}") #
                traceback.print_exc()
                if figures_dir: # Try to save an error placeholder
                    plt.figure(figsize=(8,6)); plt.text(0.5, 0.5, f"SHAP Calculation Error:\n{e_shap_calc}", ha='center', va='center');
                    try: plt.savefig(figures_dir / f"{file_prefix}_shap_calc_error.png", dpi=100);
                    except Exception: pass
                    plt.close()
                return {"error": f"SHAP calculation failed: {e_shap_calc}"}

        if shap_values is None:
            print("  ERROR: SHAP values are None after calculation attempt.")
            return {"error": "SHAP values are None."}

        # --- Process SHAP Values for Plotting (Binary vs Multiclass) ---
        shap_values_for_plot = None
        expected_value_for_plot = None

        if hasattr(explainer_obj, 'expected_value'):
            expected_value_for_plot = explainer_obj.expected_value
        elif hasattr(explainer_obj, 'base_values'): # New SHAP API
            expected_value_for_plot = explainer_obj.base_values

        if is_binary:
            print(f"  Processing binary SHAP values. Expected value type: {type(expected_value_for_plot)}")
            if isinstance(shap_values, list) and len(shap_values) == 2: # Standard output for binary proba
                shap_values_for_plot = shap_values[1] # Positive class
                if isinstance(expected_value_for_plot, (list, np.ndarray)) and len(expected_value_for_plot) == 2:
                    expected_value_for_plot = expected_value_for_plot[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2: # Already for positive class
                shap_values_for_plot = shap_values
                if isinstance(expected_value_for_plot, (list, np.ndarray)) and len(expected_value_for_plot) == 1: # if expected_value is array of one value
                    expected_value_for_plot = expected_value_for_plot[0]
            else:
                print(f"  Warning: Unexpected SHAP values format for binary. Type: {type(shap_values)}, Shape: {getattr(shap_values, 'shape', 'N/A')}")
                shap_values_for_plot = shap_values # Fallback
            if isinstance(expected_value_for_plot, np.ndarray): # If expected_value is an array (e.g. for each sample)
                expected_value_for_plot = expected_value_for_plot.mean() # Take the mean
        else: # Multiclass
            print(f"  Processing multiclass SHAP values. Expected value type: {type(expected_value_for_plot)}")
            if isinstance(shap_values, list) and len(shap_values) > class_index:
                shap_values_for_plot = shap_values[class_index]
                if isinstance(expected_value_for_plot, (list, np.ndarray)) and len(expected_value_for_plot) > class_index:
                    expected_value_for_plot = expected_value_for_plot[class_index]
                elif isinstance(expected_value_for_plot, np.ndarray): # Take mean if it's per-sample for a class
                        expected_value_for_plot = expected_value_for_plot.mean()
            else:
                print(f"  Warning: class_index {class_index} invalid for SHAP values list or unexpected format. Using raw SHAP values.")
                shap_values_for_plot = shap_values # Fallback
                if isinstance(expected_value_for_plot, np.ndarray):
                        expected_value_for_plot = expected_value_for_plot.mean()

        if expected_value_for_plot is None:
            print("  Warning: Could not determine expected_value for plot. Using 0.")
            expected_value_for_plot = 0.0
        if isinstance(expected_value_for_plot, (list, np.ndarray)): # If still array, take mean
            expected_value_for_plot = np.mean(expected_value_for_plot)


        # --- Create SHAP Explanation Object ---
        explanation = None
        try:
            explanation = shap.Explanation(
                values=shap_values_for_plot,
                base_values=np.full(shap_values_for_plot.shape[0], expected_value_for_plot) if shap_values_for_plot is not None else None,
                data=X_explain, # Use the (potentially subsampled) DataFrame
                feature_names=X_explain.columns.tolist() # Get names from DataFrame
            )
            print("  SHAP Explanation object created for plotting.")
        except Exception as e_expl:
            print(f"  ERROR Creating SHAP Explanation for plotting: {e_expl}")
            traceback.print_exc()
            if figures_dir:
                    plt.figure(figsize=(8,6)); plt.text(0.5, 0.5, f"SHAP Explanation Error:\n{e_expl}", ha='center', va='center');
                    try: plt.savefig(figures_dir / f"{file_prefix}_explanation_error_class{class_index}.png", dpi=100); plt.close(plt.gcf())
                    except Exception: pass
                    plt.close()
            return {"error": f"Failed to create Explanation for plotting: {e_expl}"}

        # --- Generate and Save Plot ---
        matplotlib_figure_created = False
        try:
            if plot_type == 'force':
                if sample_idx is None: sample_idx = 0
                if sample_idx >= explanation.values.shape[0]: sample_idx = 0
                print(f"    Generating force plot for sample {sample_idx} (HTML)...")
                html_filename = f"{file_prefix}_force_sample{sample_idx}_class{class_index}.html"
                if figures_dir:
                    save_path = figures_dir / html_filename
                    try:
                        force_plot_obj = shap.force_plot(
                            explanation.base_values[sample_idx], # Should be scalar now
                            explanation.values[sample_idx,:],    # 1D array for the sample
                            explanation.data.iloc[sample_idx,:], # Pandas Series for the sample data
                            feature_names=explanation.feature_names
                        )
                        if force_plot_obj:
                            shap.save_html(str(save_path), force_plot_obj)
                            print(f"  Saved SHAP force plot (HTML) to {save_path}")
                        else: print("  Warning: Force plot object is None.")
                    except Exception as html_e:
                        print(f"  ERROR saving force plot as HTML: {html_e}")
                else: print("  Warning: No figures_dir. Skipping force plot save.")
                return # Force plot handling is separate

            # --- Other Matplotlib-based Plots ---
            fig = plt.figure(figsize=(10, 8)) # Adjust as needed, smaller than (12,8)
            matplotlib_figure_created = True

            if plot_type == 'summary':
                shap.summary_plot(shap_values_for_plot, X_explain, feature_names=explanation.feature_names, max_display=max_display, show=False)
                filename = f"{file_prefix}_summary_class{class_index}.png"
            elif plot_type == 'bar':
                shap.plots.bar(explanation, max_display=max_display, show=False)
                filename = f"{file_prefix}_bar_class{class_index}.png"
            elif plot_type == 'beeswarm':
                shap.plots.beeswarm(explanation, max_display=max_display, show=False)
                filename = f"{file_prefix}_beeswarm_class{class_index}.png"
            elif plot_type == 'waterfall':
                if sample_idx is None: sample_idx = 0
                if sample_idx >= explanation.values.shape[0]: sample_idx = 0
                shap.plots.waterfall(explanation[sample_idx], max_display=max_display, show=False)
                filename = f"{file_prefix}_waterfall_sample{sample_idx}_class{class_index}.png"
            elif plot_type == 'grouped_waterfall':
                if sample_idx is None: sample_idx = 0
                if sample_idx >= explanation.values.shape[0]: sample_idx = 0
                print(f"    Generating grouped waterfall plot for sample {sample_idx} >= Shape of expl value {explanation.values.shape[0]}")
                sample_shap_values = explanation.values[sample_idx]
                sample_data_values = explanation.data.iloc[sample_idx].values # as numpy for easier slicing

                # Determine the number of groups based on available data
                actual_num_groups = len(feature_names) // values_per_group if values_per_group > 0 else 0
                # actual_num_groups = min(num_feature_groups, len(feature_names) // values_per_group)
                # # actual_num_groups = min(num_feature_groups, len(current_feature_names) // values_per_group if values_per_group > 0 else num_feature_groups)
                
                # Use provided metric names if they are valid
                if group_metrics and len(group_metrics) >= actual_num_groups:
                    group_names_for_plot = group_metrics[:actual_num_groups]
                    print(f"  Using provided metric names for groups: {group_names_for_plot}")
                else:
                    # Fallback to generic names if group_metrics is invalid
                    print(f"  Warning: 'group_metrics' invalid or missing. Using generic group names.")
                    group_names_for_plot = [f'F{i+1}' for i in range(actual_num_groups)]
                
                # group_names_for_plot = (group_metrics[:actual_num_groups] if group_metrics and len(group_metrics) >= actual_num_groups
                #                     else [f'Group_{i+1}' for i in range(actual_num_groups)])
                
                grouped_shap_values_agg = []
                grouped_data_repr_agg = []
                total_feats_in_sample = len(sample_shap_values)

                for i in range(actual_num_groups):
                    start_idx = i * values_per_group
                    end_idx = min(start_idx + values_per_group, total_feats_in_sample)
                    if start_idx >= total_feats_in_sample: break
                    grouped_shap_values_agg.append(np.sum(sample_shap_values[start_idx:end_idx]))
                    grouped_data_repr_agg.append(np.mean(sample_data_values[start_idx:end_idx]))
                
                remaining_start = actual_num_groups * values_per_group
                if remaining_start < total_feats_in_sample:
                    group_names_for_plot.append('Other Features')
                    grouped_shap_values_agg.append(np.sum(sample_shap_values[remaining_start:]))
                    grouped_data_repr_agg.append(np.mean(sample_data_values[remaining_start:]))

                if grouped_shap_values_agg: # Check if any groups were actually formed
                    grouped_expl = shap.Explanation(
                        values=np.array(grouped_shap_values_agg),
                        base_values=explanation.base_values[sample_idx] if isinstance(explanation.base_values, np.ndarray) else explanation.base_values,
                        data=np.array(grouped_data_repr_agg),
                        feature_names=group_names_for_plot
                    )
                    shap.plots.waterfall(grouped_expl, max_display=max_display, show=False)
                    filename = f"{file_prefix}_grouped_waterfall_sample{sample_idx}_class{class_index}.png"
                else:
                    print("    Skipping grouped waterfall: no feature groups formed.")
                    filename = ""
            
            elif plot_type == 'circular_heatmap':
                print("    Preparing data for circular heatmap...")
                print(f"Group metrics: {group_metrics}, values_per_group: {values_per_group}")
                print(f"shap_values_for_plot type: {type(shap_values_for_plot)}, shape: {getattr(shap_values_for_plot, 'shape', 'N/A')}")
                # if shap_values_for_plot is not None and group_metrics and values_per_group:
                if group_metrics and values_per_group > 0:
                    # The plot_circular_shap_heatmap function creates and saves its own figure
                    plot_circular_shap_heatmap(
                        shap_values=shap_values_for_plot,
                        feature_names=group_metrics,
                        num_zones_per_metric=values_per_group,
                        model_name_prefix=model_name_str,
                        shap_output_dir=figures_dir
                    )
                else:
                    print("    Skipping circular heatmap due to missing data (SHAP values, group_metrics, or values_per_group).")
                    
                filename = ""
            
            # elif plot_type == 'circular_heatmap':
            #     print(" Preparing data for circular heatmap...")
            #     if shap_values_for_plot is not None and group_metrics and values_per_group:
            #         # custom plot function might create and save its own figure
            #         # So we prevent the main saving logic from running by setting filename to ""
            #         plot_circular_shap_heatmap(
            #             shap_values=shap_values_for_plot,
            #             feature_names=group_metrics,
            #             num_zones_per_metric=values_per_group,
            #             model_name_prefix=model_name_str,
            #             shap_output_dir=figures_dir
            #         )
                # else:
                #     print("Skipping circular heatmap > missing data (SHAP values, group_metrics, or values_per_group).")
                # filename = "" 
            
            else:
                print(f"  Warning: Unknown plot_type '{plot_type}' for Matplotlib.")
                filename = ""

            if filename: # If a plot was supposed to be generated
                plt.title(f'SHAP {plot_type.replace("_", " ").title()} - {model_name_str} (Class {class_index})', fontsize=10)
                plt.tight_layout(pad=1.5) # Add some padding
                if figures_dir:
                    save_path = figures_dir / filename
                    print(f"    Attempting to save: {save_path}")
                    plt.savefig(save_path, dpi=200, bbox_inches='tight') # Reduced DPI slightly
                    plt.close(plt.gcf()) # Close the figure after saving
                    print(f"  Saved SHAP plot to {save_path}")
                else:
                    print("  Warning: No figures_dir. Skipping plot save.")
        
        except ValueError as ve:
                print(f"  ValueError during plot generation/saving '{filename}': {ve}")
                if "Image size" in str(ve) and figures_dir and filename:
                    try:
                        fallback_path = (figures_dir / filename).with_suffix('.jpg')
                        print(f"    Attempting fallback save to JPG: {fallback_path}")
                        plt.savefig(fallback_path, dpi=100, bbox_inches='tight', quality=80)
                        plt.close(plt.gcf())
                        print(f"    Saved as fallback JPG.")
                    except Exception as e_fallback:
                        print(f"    Fallback save also failed: {e_fallback}")
        except Exception as plot_e:
            print(f"  Error generating/saving SHAP plot '{plot_type}': {plot_e}")
            traceback.print_exc()
        finally:
            if matplotlib_figure_created:
                plt.close(plt.gcf())

        # --- Coefficient Importance Plots (if bar plot and linear model) ---
        if plot_type == 'bar' and isinstance(base_model, (LogisticRegression, LinearRegression, Ridge, Lasso, SVC)):
            if hasattr(base_model, 'coef_') and (not isinstance(base_model, SVC) or base_model.kernel == 'linear'):
                print(f"  Generating coefficient importance plots for {model_name_str} (Class {class_index})...")
                try:
                    coeffs_all = base_model.coef_
                    coeffs_for_class = None
                    if len(coeffs_all.shape) == 1: # Binary (n_features,) or single output
                        coeffs_for_class = coeffs_all
                    elif coeffs_all.shape[0] == 1: # Binary (1, n_features)
                        coeffs_for_class = coeffs_all[0]
                    elif coeffs_all.shape[0] > class_index: # Multiclass
                        coeffs_for_class = coeffs_all[class_index]
                    else:
                        print(f"    Warning: class_index {class_index} out of bounds for coefficients. Using class 0.")
                        coeffs_for_class = coeffs_all[0]

                    if coeffs_for_class.shape[0] != len(current_feature_names):
                        print(f"    Warning: Coeff shape {coeffs_for_class.shape} mismatch with names {len(current_feature_names)}. Skipping.")
                    else:
                        abs_coeffs = np.abs(coeffs_for_class)
                        sorted_indices = np.argsort(abs_coeffs)[::-1][:max_display]

                        plt.figure(figsize=(10, 8))
                        plt.barh(range(len(sorted_indices)), abs_coeffs[sorted_indices], align='center')
                        plt.yticks(range(len(sorted_indices)), [current_feature_names[i] for i in sorted_indices])
                        plt.gca().invert_yaxis()
                        plt.xlabel('Absolute Coefficient Magnitude')
                        plt.title(f'{model_name_str} Top {max_display} Coefficients (Class {class_index})')
                        plt.tight_layout()
                        coef_filename = f"{file_prefix}_coefficient_importance_class{class_index}.png"
                        if figures_dir:
                            plt.savefig(figures_dir / coef_filename, dpi=200, bbox_inches='tight')
                            print(f"    Saved coefficient plot to {figures_dir / coef_filename}")
                            plt.close(plt.gcf())
                except Exception as e_coeff:
                    print(f"    Error generating coefficient plots: {e_coeff}")
                    if plt.get_fignums(): plt.close(plt.gcf()) # Close if fig was created
    

    except Exception as e_outer:
        print(f"\n--- Outer Error during SHAP analysis for {model_name_str} ---")
        print(f"Error message: {e_outer}")
        traceback.print_exc()
        if plt.get_fignums(): plt.close('all') # Close any open figures
        return {"error": str(e_outer)}


def run_shap_analysis_eval_v1(model, X_test, feature_names, plot_type, class_index, sample_idx=0, report_dir=None, group_metrics=None, values_per_group=20, **kwargs):
    """
    A robust, self-contained function that correctly handles various SHAP explainer
    outputs for binary and multiclass models before generating plots.
    """
    model_name_str = get_standardized_model_name(model)
    logger.info(f"--- Running SHAP Analysis (Plot: {plot_type}, Class: {class_index}, Model: {model_name_str}) ---")
    
    try:
        # --- 1. Standardize Data & Get Explainer ---
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        base_model = model.steps[-1][1] if isinstance(model, Pipeline) else model
        
        if isinstance(base_model, (RandomForestClassifier, GradientBoostingClassifier, xgb.XGBClassifier, LogisticRegression)):
            explainer = shap.Explainer(model, X_test_df)
        else:
            logger.info("    Using KernelExplainer as fallback.")
            background_data = shap.sample(X_test_df, min(100, X_test_df.shape[0]))
            explainer = shap.KernelExplainer(model.predict_proba, background_data)
        
        shap_values = explainer.shap_values(X_test_df)

        # --- 2. THE FIX: Correctly process different SHAP value structures ---
        shap_values_for_class = None
        base_value_for_class = None

        if isinstance(shap_values, list): # KernelExplainer binary or any multiclass
            shap_values_for_class = shap_values[class_index]
            base_value_for_class = explainer.expected_value[class_index]
        else: # Linear/Tree binary case (single 2D array)
            shap_values_for_class = shap_values if class_index == 1 else -shap_values
            base_value_for_class = explainer.expected_value if class_index == 1 else (1 - explainer.expected_value)

        # --- 3. Create a clean Explanation object for plotting ---
        final_explanation = shap.Explanation(
            values=shap_values_for_class,
            base_values=np.full(X_test_df.shape[0], base_value_for_class),
            data=X_test_df.values,
            feature_names=X_test_df.columns.tolist()
        )
        
        # --- 4. Generate and Save Plot ---
        plt.figure(figsize=(10, 8))
        if plot_type == 'summary':
            shap.summary_plot(final_explanation, show=False)
        elif plot_type == 'bar':
            shap.plots.bar(final_explanation, show=False)
        elif plot_type == 'waterfall':
            shap.plots.waterfall(final_explanation[sample_idx], show=False)
        elif plot_type == 'grouped_waterfall':
            if not group_metrics:
                logger.warning("Skipping grouped_waterfall: group_metrics not provided.")
                plt.close()
                return
            instance_expl = final_explanation[sample_idx]
            grouped_vals = [instance_expl.values[i*values_per_group:(i+1)*values_per_group].sum() for i in range(len(group_metrics))]
            grouped_expl = shap.Explanation(values=np.array(grouped_vals), base_values=instance_expl.base_values, feature_names=group_metrics)
            shap.plots.waterfall(grouped_expl, show=False)

        filename = f"{model_name_str}_shap_{plot_type}_class{class_index}"
        if plot_type in ['waterfall', 'grouped_waterfall']:
            filename += f"_sample{sample_idx}"
        
        if report_dir:
            save_path = Path(report_dir) / "figures"; save_path.mkdir(exist_ok=True, parents=True)
            plt.savefig(save_path / (filename + ".png"), dpi=150, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.error(f"Error in run_shap_analysis_eval for {model_name_str}: {e}", exc_info=True)

def run_shap_analysis_eval_workswithinstance(model, X_test, feature_names=None, plot_type='summary',
                           class_index=0, sample_idx=0, report_dir=None, y_test=None,
                           group_metrics=None, values_per_group=20, **kwargs):
    """
    A robust, self-contained function to calculate SHAP values and generate plots.
    It handles different model types and data formats internally to ensure compatibility.
    """
    logger.info(f"--- Running SHAP Analysis (Plot: {plot_type}, Class: {class_index}, Sample: {sample_idx}) ---")
    
    try:
        # --- 1. Setup and Data Preparation ---
        X_test_df = pd.DataFrame(X_test, columns=feature_names) if not isinstance(X_test, pd.DataFrame) else X_test.copy()
        model_name_str = get_standardized_model_name(model)
        base_model = model.steps[-1][1] if isinstance(model, Pipeline) else model
        figures_dir = Path(report_dir) / 'figures' if report_dir else None
        if figures_dir:
            figures_dir.mkdir(parents=True, exist_ok=True)

        # --- 2. Robust Explainer Initialization and Value Calculation ---
        logger.info(f"  Initializing SHAP explainer for {model_name_str}...")
        if isinstance(base_model, (RandomForestClassifier, GradientBoostingClassifier, xgb.XGBClassifier, LogisticRegression)):
            explainer = shap.Explainer(model, X_test_df)
            shap_values_raw = explainer.shap_values(X_test_df)
        else:
            logger.info("    Using KernelExplainer as fallback.")
            background_data = shap.sample(X_test_df, min(100, X_test_df.shape[0]))
            explainer = shap.KernelExplainer(model.predict_proba, background_data)
            shap_values_raw = explainer.shap_values(X_test_df)

        # --- 3. Process SHAP Values for the Target Class ---
        is_multiclass = isinstance(shap_values_raw, list)
        shap_values_for_class = None
        
        if is_multiclass:
            if class_index < len(shap_values_raw):
                shap_values_for_class = shap_values_raw[class_index]
        else:
            shap_values_for_class = shap_values_raw if class_index == 1 else -shap_values_raw

        if shap_values_for_class is None:
            logger.error("Could not extract SHAP values for the specified class.")
            return

        # --- 4. Generate Plot ---
        plt.figure(figsize=(10, 8))
        features_np = X_test_df.to_numpy()
        feature_names_list = list(X_test_df.columns)

        if plot_type == 'summary':
            shap.summary_plot(shap_values_for_class, features=features_np, feature_names=feature_names_list, show=False)
        elif plot_type == 'bar':
            shap.summary_plot(shap_values_for_class, features=features_np, feature_names=feature_names_list, plot_type='bar', show=False)
        elif plot_type == 'waterfall':
            base_value = explainer.expected_value[class_index] if is_multiclass else (explainer.expected_value if class_index == 1 else 1 - explainer.expected_value)
            temp_expl = shap.Explanation(values=shap_values_for_class[sample_idx], base_values=base_value,
                                         data=features_np[sample_idx], feature_names=feature_names_list)
            shap.plots.waterfall(temp_expl, show=False)
        
        
        # --- 5. Save Plot ---
        # FIX: Changed class_idx to class_index to match the function's argument name
        filename = f"{model_name_str}_shap_{plot_type}_class{class_index}"
        if plot_type == 'waterfall':
            filename += f"_sample{sample_idx}"
        
        if figures_dir:
            plt.savefig(figures_dir / (filename + ".png"), dpi=150, bbox_inches='tight')
        plt.close()

    except Exception as e:
        logger.error(f"Error in run_shap_analysis_eval for {model_name_str}: {e}")
        logger.error(traceback.format_exc())

# Analyze feature group importance
def analyze_feature_group_importance(model, X_test, num_features=9, values_per_feature=20, metrics=None, class_index=0, report_dir=None, y_test = None):
    """
    Analyze importance of feature groups and save the visualization
    
    Args:
        model: Trained model
        X_test: Test data
        num_features: Number of features (metrics)
        values_per_feature: Number of zones per feature
        metrics: Names of metrics
        class_index: Index of class to analyze
        report_dir: Directory to save the plot (optional)
    """
    if metrics is None:
        metrics = ['mean', 'med', 'std', 'iqr', 'idr', 'skew', 'kurt', 'Del', 'Amp']
    
    try:
        # Get standardized model name for the filename
        model_name = get_standardized_model_name(model)
        
        # Create filename with shap prefix
        filename = f"{model_name}_shap_feature_gp_importance_class{class_index}.png"
        
        # Determine if this is a tree-based model that can use TreeExplainer
        is_tree_model = isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, xgb.XGBClassifier))
        
        # Extract the base model if it's a pipeline
        base_model = model
        if isinstance(model, Pipeline):
            base_model = model.steps[-1][1]
   
        values = None # Initialize values
        base_model_name = type(base_model).__name__ # Get name of the actual model

        print(f"  Determining importance for {model_name} (Base: {base_model_name}, IsTree: {is_tree_model})...")

        if is_tree_model:
            print(f"    Calculating SHAP values using TreeExplainer...")
            try:
                explainer = shap.TreeExplainer(model) # Use original model (pipeline or base)
                shap_values_output = explainer.shap_values(X_test)

                # --- Process TreeExplainer Output ---
                if isinstance(shap_values_output, list): # Multiclass output
                    print(f"    TreeExplainer returned {len(shap_values_output)} classes).")
                    if class_index < len(shap_values_output):
                        values = np.abs(shap_values_output[class_index])
                    else:
                        print(f"    Warning: class_index {class_index} out of bounds. Using class 0.")
                        values = np.abs(shap_values_output[0])
                elif isinstance(shap_values_output, np.ndarray) and shap_values_output.ndim == 3: # Binary case as (samples, feats, classes=2)
                     print(f"    TreeExplainer returned 3D array (binary). Shape: {shap_values_output.shape}")
                     if shap_values_output.shape[2] >= max(1, class_index + 1): # Ensure class_index is valid
                         values = np.abs(shap_values_output[:, :, class_index]) # Use specified class (usually 1 for positive)
                     else:
                         print(f"    Warning: class_index {class_index} out of bounds for 3D SHAP array. Using class 0.")
                         values = np.abs(shap_values_output[:, :, 0])
                elif isinstance(shap_values_output, np.ndarray) and shap_values_output.ndim == 2: # Binary (positive class) or Regression
                    print(f"    TreeExplainer returned 2D array. Shape: {shap_values_output.shape}")
                    values = np.abs(shap_values_output)
                else:
                    print(f"    Warning: Unexpected SHAP values format from TreeExplainer: {type(shap_values_output)}")
                    values = None # Force fallback
            except Exception as tree_e:
                print(f"    TreeExplainer failed: {tree_e}. Falling back.")
                values = None # Force fallback

        # --- Fallback or Non-Tree Models ---
        if values is None: # If TreeExplainer failed or it's not a tree model
             if hasattr(base_model, 'coef_'): # Linear models
                 print(f"    Using coefficients for {base_model_name}...")
                 coefs_all = base_model.coef_
                 if len(coefs_all.shape) == 1: # Binary (n_features,)
                     values = np.abs(coefs_all)
                 elif coefs_all.shape[0] == 1: # Binary (1, n_features)
                     values = np.abs(coefs_all[0])
                 else: # Multiclass (n_classes, n_features)
                     if class_index < coefs_all.shape[0]:
                         values = np.abs(coefs_all[class_index])
                     else:
                         print(f"    Warning: class_index {class_index} out of bounds for coefficients. Using class 0.")
                         values = np.abs(coefs_all[0])

             elif hasattr(base_model, 'feature_importances_'): # Some other models
                 print(f"    Using feature_importances_ for {base_model_name}...")
                 importances = base_model.feature_importances_
                 values = np.abs(importances) # Use absolute importance

             else: # Fallback to permutation importance
                 print(f"    Model {base_model_name} has no direct importance. Using permutation importance fallback...")
                 if y_test is not None:
                     try:
                         print("      Calculating permutation importance (this might take a while)...")
                         sample_size = min(100, X_test.shape[0]) # Use a subset for speed
                         X_sample = X_test[:sample_size]
                         y_sample = y_test[:sample_size]
                         perm_results = permutation_importance(model, X_sample, y_sample,
                                n_repeats=5, random_state=42, n_jobs=1, scoring='f1_weighted')
                         importances = perm_results.importances_mean
                         values = np.abs(importances) # Use absolute importance
                         print(f"      Permutation importance calculated.")
                     except Exception as perm_e:
                         print(f"      Permutation importance failed: {perm_e}. Using fallback.")
                         values = None # Force last fallback
                 else:
                     print("      Cannot calculate permutation importance (y_test not provided). Using fallback.")
                     values = None # Force last fallback

                 if values is None: # Last resort fallback
                     print("      Using uniform importance as last resort.")
                     values = np.ones(X_test.shape[1]) * 0.1 # Small non-zero uniform importance

        # --- Ensure 'values' is 2D (samples, features) or 1D (features,) ---
        if values is None: # Should not happen with fallbacks, but check
             print(f"  ERROR: Could not determine feature importance values for {model_name}. Aborting group analysis.")
             return {}
        if values.ndim == 1: # If we got a 1D array (e.g., from coef_, feature_importances_, permutation)
             print(f"    Expanding 1D importance array ({values.shape}) to 2D for processing.")
             values = np.tile(values, (X_test.shape[0], 1)) # Tile to match X_test samples dimension

        
        
        # # Add a special case for StackingClassifier
        # if isinstance(model, StackingClassifier):
        #     print("Special handling for StackingClassifier feature importance")
        #     try:
        #         # Try to use one of the base estimators for feature importance
        #         if hasattr(model, 'estimators_') and len(model.estimators_) > 0:
        #             # Find a base estimator with feature_importances_ if possible
        #             for estimator in model.estimators_:
        #                 if hasattr(estimator, 'feature_importances_'):
        #                     print(f"Using {type(estimator).__name__} base estimator for feature importance")
        #                     importances = estimator.feature_importances_
        #                     values = np.tile(importances, (X_test.shape[0], 1))
        #                     break
        #             else:
        #                 # No suitable base estimator found, fall back to permutation importance
        #                 raise ValueError("No base estimator with feature_importances_ found")
        #         else:
        #             raise ValueError("No base estimators available")
        #     except:
        #         # If that fails, use permutation importance
        #         try:
        #             # Use a subset of data for faster computation
        #             sample_size = min(200, X_test.shape[0])
        #             result = permutation_importance(model, X_test[:sample_size], 
        #                                         model.predict(X_test[:sample_size]),
        #                                         n_repeats=5, random_state=42)
        #             importances = result.importances_mean
        #             values = np.tile(importances, (X_test.shape[0], 1))
        #         except Exception as e:
        #             print(f"Permutation importance failed: {e}")
        #             # Add slight random variation to uniform importance
        #             values = np.ones((X_test.shape[0], X_test.shape[1]))
        #             values = values * (1 + np.random.uniform(-0.2, 0.2, values.shape))
        
        # Calculate group importance
        group_importance = {}
        
        # Determine actual number of features based on values shape and values_per_feature
        total_features = values.shape[1]
        actual_num_features = min(num_features, total_features // values_per_feature)
        
        print(f"Analyzing feature group importance for {actual_num_features} features with {values_per_feature} values each")
        
        # Use actual metrics if provided, otherwise use default names but limit to actual_num_features
        feature_names = metrics[:actual_num_features] if len(metrics) >= actual_num_features else [f'Feature{i}' for i in range(actual_num_features)]
        
        for i in range(actual_num_features):
            start_idx = i * values_per_feature
            end_idx = min(start_idx + values_per_feature, total_features)  # Ensure we don't exceed bounds
            
            # Get the importance for this feature group
            importance = np.abs(values[:, start_idx:end_idx]).mean()
            
            # Use provided metric name if available, otherwise use generic name
            metric_name = feature_names[i] if i < len(feature_names) else f'Feature{i}'
            group_importance[metric_name] = float(importance)  # Convert to float to ensure it's serializable
        
        # Plot group importance
        plt.figure(figsize=(10, 6))
        sorted_importance = {k: v for k, v in sorted(group_importance.items(), key=lambda item: item[1], reverse=True)}
        
        # plt.bar(sorted_importance.keys(), sorted_importance.values())
        bars = plt.bar(sorted_importance.keys(), sorted_importance.values()) #, color='skyblue')
        # Add data labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(
                x=bar.get_x() + bar.get_width() / 2, 
                y=height, 
                s=f'{height:.4f}', # 
                ha='center', 
                va='bottom',
                # rotation=90, # Rotate
                fontsize=10
            )
        
        
        plt.title(f'Feature Group Importance - {model_name} (Class {class_index})')
        plt.xticks(rotation=45)
        plt.ylabel('Mean |Importance|')
        plt.tight_layout(pad=1.0)
        
        # Save the figure if report_dir is provided
        if report_dir:
            try:
                from pathlib import Path
                figures_dir = Path(report_dir) / 'figures'
                figures_dir.mkdir(parents=True, exist_ok=True)
                
                save_path = figures_dir / filename
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Saved feature group importance plot to {save_path}")
                plt.close(plt.gcf())
            except Exception as e:
                print(f"Error saving feature group importance plot: {e}")
                traceback.print_exc()
        
        # Only show in interactive mode
        if plt.isinteractive():
            plt.show()
        else:
            plt.close()
        
        return group_importance
        
    except Exception as e:
        print(f"Error in feature group importance analysis: {str(e)}")
        traceback.print_exc()
        return {}




def analyze_feature_group_zones(model, X_test, feature_names, feature_indices=None, group_name="med", report_dir=None, class_index=0, plot_type='bar'):
    """
    Create a visualization showing the importance of each zone for a specific feature group
    with model name included in the filename.
    
    Args:
        model: Trained model
        X_test: Test data
        feature_names: List of feature names
        feature_indices: Optional list of indices to use
        group_name: Specific feature group to analyze (e.g., "med")
        report_dir: Directory to save visualization
        class_index: Class index for multi-class models
        plot_type: Type of plot to generate ('bar', 'heatmap', or 'clustermap')
    """
    try:
        # Get the model name
        if hasattr(model, '__class__'):
            model_type = model.__class__.__name__
        else:
            model_type = "Unknown"
            
        # For pipelines, get the last step (actual model)
        if hasattr(model, 'steps') and len(model.steps) > 0:
            model_type = model.steps[-1][1].__class__.__name__
            
        # Clean up model type for filename
        model_name = model_type.replace(" ", "_")
        
        print(f"\nAnalyzing zone importance for feature group '{group_name}' in {model_name}...")
        
        # for feature indexing, i.e. features < 9
        # Generate feature names if not provided or if mismatch with X_test
        if feature_names is None or len(feature_names) != X_test.shape[1]:
            metrics = ['mean', 'med', 'std', 'iqr', 'idr', 'skew', 'kurt', 'Del', 'Amp']
            num_zones = 20
            
            if feature_indices is not None:
                # Generate feature names only for the selected feature indices
                selected_feature_names = []
                for idx in feature_indices:
                    if idx < len(metrics):  # Ensure idx is valid
                        metric = metrics[idx]
                        # Add all zones for this metric
                        for zone in range(1, num_zones + 1):
                            selected_feature_names.append(f'{metric}_Z{zone}')
                
                feature_names = selected_feature_names
            else:
                # Generate all feature names
                feature_names = [f'{m}_Z{i+1}' for m in metrics for i in range(num_zones)]
            
            # Ensure feature_names matches X_test.shape[1]
            if len(feature_names) != X_test.shape[1]:
                print(f"Warning: Generated {len(feature_names)} feature names but X_test has {X_test.shape[1]} columns")
                feature_names = [f'feaT_{i}' for i in range(X_test.shape[1])]
        # ------------ end-------------
        
        # Get feature importances safely
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            print(f"Using feature_importances_ from {model_name} ({len(importances)} values)")
        elif hasattr(model, 'coef_'):
            # Linear models
            if len(model.coef_.shape) == 1:
                # Binary classification - single coefficient vector
                importances = np.abs(model.coef_)
                print(f"Using coefficient magnitudes for binary {model_name} ({len(importances)} values)")
            else:
                # Multi-class - handle the coef shape carefully
                if class_index >= model.coef_.shape[0]:
                    print(f"Warning: class_index {class_index} >= Model.coef_.shape[0] = {model.coef_.shape[0]}, out of bounds for {model_name}. Using class 0 instead. ")
                    class_index = 0
                importances = np.abs(model.coef_[class_index])
                print(f"Using coefficient magnitudes for class {class_index} in {model_name} ({len(importances)} values)")
        else:
            print(f"Model {model_name} doesn't have feature importance attribute. Skipping zone analysis! ")
            return None
        
        # Check if lengths match
        if len(importances) != len(feature_names):
            print(f"Warning: Length mismatch - importances: {len(importances)}, feature_names: {len(feature_names)}")
            # Adjust lengths to match
            if len(importances) > len(feature_names):
                importances = importances[:len(feature_names)]
            else:
                # Pad importances with zeros
                importances = np.pad(importances, (0, len(feature_names) - len(importances)))
        
        # Filter features for the specified group
        group_features = []
        group_importances = []
        zone_numbers = []
        
        for i, feature in enumerate(feature_names):
            parts = feature.split('_')
            if len(parts) > 1 and parts[0] == group_name:
                # Extract zone number for sorting
                try:
                    # Handle formats like "med_Z5" - extract the "5"
                    zone_str = parts[1]
                    # Remove 'Z' prefix if present
                    if zone_str.startswith('Z'):
                        zone_str = zone_str[1:]
                    zone_num = int(zone_str)
                    
                    group_features.append(feature)
                    group_importances.append(importances[i])
                    zone_numbers.append(zone_num)
                except (ValueError, IndexError):
                    # Skip features with invalid naming
                    continue
        
        if not group_features:
            print(f"No features found for group '{group_name}' in {model_name}")
            return None
        
        # Sort by zone number for better visualization
        sorted_indices = np.argsort(zone_numbers)
        sorted_features = [group_features[i] for i in sorted_indices]
        sorted_importances = [group_importances[i] for i in sorted_indices]
        sorted_zone_nums = [zone_numbers[i] for i in sorted_indices]
        
        # Different visualization types
        if plot_type == 'bar':
            # Create standard bar plot
            plt.figure(figsize=(8, 6), dpi=100)
            
            # Create x-axis labels with just zone numbers
            x_labels = [f"Zone {z}" for z in sorted_zone_nums]
            
            # Create bar plot
            plt.bar(range(len(sorted_importances)), sorted_importances)
            plt.xticks(range(len(sorted_importances)), x_labels, rotation=45, ha='right')
            plt.title(f'{model_name}: {group_name} Feature Importance by Zone (Class {class_index})')
            plt.xlabel('Zone')
            plt.ylabel('Importance')
            
            # Set tight layout with small margins
            plt.tight_layout(pad=1.0)
            
            # File name suffix for saving
            plot_suffix = "bar"
            
        elif plot_type == 'heatmap' or plot_type == 'clustermap':
            # For heatmap/clustermap, we need to organize all metrics and zones
            
            # Get all unique metrics and their zones from feature names
            all_metrics = set()
            all_metrics_data = {}
            
            for i, feature in enumerate(feature_names):
                parts = feature.split('_')
                if len(parts) > 1 and parts[0] != '':
                    metric = parts[0]
                    all_metrics.add(metric)
                    
                    # Extract zone number
                    try:
                        zone_str = parts[1]
                        if zone_str.startswith('Z'):
                            zone_str = zone_str[1:]
                        zone_num = int(zone_str)
                        
                        if metric not in all_metrics_data:
                            all_metrics_data[metric] = {}
                        
                        all_metrics_data[metric][zone_num] = importances[i]
                    except (ValueError, IndexError):
                        continue
            
            # Find the maximum zone number across all metrics
            max_zone = max([max(zones.keys()) for metric, zones in all_metrics_data.items() if zones])
            
            # Create a complete matrix for all metrics and zones
            metric_list = sorted(list(all_metrics))
            heatmap_data = np.zeros((len(metric_list), max_zone))
            
            # Fill the matrix with importance values
            for i, metric in enumerate(metric_list):
                if metric in all_metrics_data:
                    for zone, importance in all_metrics_data[metric].items():
                        if 1 <= zone <= max_zone:  # Ensure zone is valid
                            heatmap_data[i, zone-1] = importance
            
            # For the specific group, highlight it in the visualization
            group_index = -1
            if group_name in metric_list:
                group_index = metric_list.index(group_name)
            
            if plot_type == 'heatmap':
                # Create heatmap
                plt.figure(figsize=(max(8, max_zone * 0.5), max(6, len(metric_list) * 0.4)))
                hm = sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="viridis",
                              yticklabels=metric_list,
                              xticklabels=[f"Z{z+1}" for z in range(max_zone)])
                
                plt.title(f'{model_name}: Feature Importance by Zone (Class {class_index})')
                plt.ylabel("Metric Group")
                plt.xlabel("Zone")
                
                # If specific group was requested, highlight it
                if group_index >= 0:
                    # Add a rectangle around the row for the specific group
                    hm.add_patch(plt.Rectangle((0, group_index), max_zone, 1, fill=False, 
                                           edgecolor='red', lw=2))
                
                plot_suffix = "heatmap"
                
            elif plot_type == 'clustermap':
                # Create clustermap
                hm_cluster = sns.clustermap(heatmap_data, annot=True, fmt=".3f", cmap="viridis",
                                       yticklabels=metric_list,
                                       xticklabels=[f"Z{z+1}" for z in range(max_zone)],
                                       figsize=(max(8, max_zone * 0.6), max(6, len(metric_list) * 0.5)))
                
                plt.title(f'{model_name}: Clustered Feature Importance by Zone (Class {class_index})')
                
                
                plot_suffix = "clustermap"
        else:
            print(f"Unsupported plot_type: {plot_type}. Using default bar plot.")
            plot_type = 'bar'
            plot_suffix = "bar"
            
            # Create default bar plot
            plt.figure(figsize=(8, 6), dpi=100)
            x_labels = [f"Zone {z}" for z in sorted_zone_nums]
            plt.bar(range(len(sorted_importances)), sorted_importances)
            plt.xticks(range(len(sorted_importances)), x_labels, rotation=45, ha='right')
            plt.title(f'{model_name}: {group_name} Feature Importance by Zone (Class {class_index})')
            plt.xlabel('Zone')
            plt.ylabel('Importance')
            plt.tight_layout(pad=1.0)
        
        # Save figure safely
        if report_dir:
            try:
                figures_dir = Path(report_dir) / 'figures'
                figures_dir.mkdir(parents=True, exist_ok=True)
                
                # Use filename with model name and plot type
                filename = f"{model_name}_shap_{group_name}_zones_class{class_index}_{plot_suffix}.png"
                save_path = figures_dir / filename
                
                # Save with low DPI to avoid image size issues
                plt.savefig(save_path, dpi=100, bbox_inches='tight')
                print(f"Saved {model_name}_{group_name} zone {plot_type} to {save_path}")
            except Exception as save_err:
                print(f"Error saving visualization: {save_err}")
        
        # Close the figure to avoid memory leaks
        plt.close(plt.gcf())
        
        # Return the sorted data for further analysis
        result = pd.DataFrame({
            'zone': sorted_zone_nums,
            'feature': sorted_features,
            'importance': sorted_importances
        })
        
        # Print top zones
        print(f"\nTop 5 zones by importance for {model_name} ({group_name} features):")
        for i in range(min(5, len(result))):
            print(f"  Zone {result['zone'].iloc[i]}: {result['importance'].iloc[i]:.6f}")
        
        return result
        
    except Exception as e:
        print(f"Error in analyze_feature_group_zones: {e}")
        import traceback
        traceback.print_exc()
        return None









def create_feature_transformations_fit(X, feature_names=None, num_features=9, values_per_feature=20):
    """
    Calculate transformation parameters from training data
    
    Args:
        X: Input features (training data only)
        feature_names: Names of features
        num_features: Number of feature groups
        values_per_feature: Number of values per feature
    
    Returns:
        params: Dictionary of transformation parameters
        new_feature_names: Names of transformed features
    """
    # Calculate transformation parameters from training data
    params = {}
    new_feature_names = feature_names.copy() if feature_names is not None else None
    
    # Calculate mean for each feature group
    means = []
    for i in range(num_features):
        start_idx = i * values_per_feature
        end_idx = start_idx + values_per_feature
        group_mean = np.nanmean(X[:, start_idx:end_idx], axis=1)
        means.append(group_mean)
    
    # Store means in parameters
    params['means'] = np.column_stack(means)
    
    # Add names for new mean features
    if feature_names is not None:
        feature_group_names = [name.split('_')[0] for name in feature_names[::values_per_feature]]
        mean_feature_names = [f"{name}_mean" for name in feature_group_names]
        new_feature_names.extend(mean_feature_names)
    
    # Calculate standard deviation for each feature group
    stds = []
    for i in range(num_features):
        start_idx = i * values_per_feature
        end_idx = start_idx + values_per_feature
        group_std = np.nanstd(X[:, start_idx:end_idx], axis=1)
        stds.append(group_std)
    
    # Store stds in parameters
    params['stds'] = np.column_stack(stds)
    
    # Add names for new std features
    if feature_names is not None:
        std_feature_names = [f"{name}_std" for name in feature_group_names]
        new_feature_names.extend(std_feature_names)
    
    # Store feature group names for later use
    params['feature_group_names'] = feature_group_names if feature_names is not None else None
    
    # Store polynomial feature generator if applicable
    if not np.isnan(params['means']).any():
        try:
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            # Fit the polynomial transformer but don't transform yet
            poly.fit(params['means'])
            params['poly'] = poly
            
            if feature_names is not None:
                # Create names for polynomial features
                poly_feature_names = []
                feature_indices = poly.get_feature_indices(n_input_features=params['means'].shape[1])
                powers = poly.powers_[params['means'].shape[1]:]  # Skip the original features
                
                for power in powers:
                    indices = np.where(power > 0)[0]
                    if len(indices) > 1:  # Only include interaction terms
                        term_name = "*".join([feature_group_names[i] for i in indices])
                        poly_feature_names.append(f"{term_name}_poly")
                
                new_feature_names.extend(poly_feature_names)
        except Exception as e:
            print(f"Warning: Error creating polynomial features: {e}")
            params['poly'] = None
    else:
        print("Warning: NaN values detected in means, skipping polynomial features")
        params['poly'] = None
    
    return params, new_feature_names

def create_feature_transformations_transform(X, params, feature_names=None, num_features=9, values_per_feature=20):
    """
    Apply transformation using pre-computed parameters
    
    Args:
        X: Input features (can be training or test data)
        params: Dictionary of transformation parameters
        feature_names: Names of features
        num_features: Number of feature groups
        values_per_feature: Number of values per feature
    
    Returns:
        X_transformed: Transformed features
    """
    # Create a copy of original features
    X_transformed = X.copy()
    
    # Add mean features
    X_transformed = np.hstack((X_transformed, params['means']))
    
    # Add std features
    X_transformed = np.hstack((X_transformed, params['stds']))
    
    # Create interaction terms between the means of different feature groups
    interactions = []
    for i in range(num_features):
        for j in range(i+1, num_features):
            # Multiply means of two different feature groups
            interaction = params['means'][:, i] * params['means'][:, j]
            interactions.append(interaction)
    
    if interactions:
        interactions = np.column_stack(interactions)
        X_transformed = np.hstack((X_transformed, interactions))
    
    # Add polynomial features if available
    if params['poly'] is not None:
        poly_features = params['poly'].transform(params['means'])
        # Only keep the interaction terms
        if poly_features.shape[1] > params['means'].shape[1]:
            poly_features = poly_features[:, params['means'].shape[1]:]
            X_transformed = np.hstack((X_transformed, poly_features))
    
    # Final check for NaN values in the transformed features
    if np.isnan(X_transformed).any():
        print(f"Warning: {np.isnan(X_transformed).sum()} NaN values in transformed features. Replacing with feature means...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_transformed = imputer.fit_transform(X_transformed)
    
    return X_transformed


def evaluate_model_with_cv(model, X, y, model_name, n_splits=3, random_state=42):
    """
    Evaluate model using stratified k-fold cross-validation with automatic fold adjustment
    
    Args:
        model: Trained model
        X: Features
        y: Target variable
        model_name: Name of the model
        n_splits: Maximum number of cross-validation splits
        random_state: Random state for reproducibility
    
    Returns:
        cv_results: Dictionary with cross-validation results
    """
    from sklearn.model_selection import StratifiedKFold, cross_validate, LeaveOneOut
    from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score
    import numpy as np
  
    # Check if n_splits is too large for the minority class
    class_counts = np.bincount(y)
    min_samples = min(class_counts[class_counts > 0])  # Ignore classes with zero samples
    
    # original_cv = 5
    # # Determine appropriate number of folds
    # if min_samples < 5:  # Small but not tiny class
    #     adjusted_cv = max(2, min(3, min_samples))
    #     print(f"Adjusting grid search: Minority class has only {min_samples} samples, "
    #         f"reducing folds from {original_cv} to {adjusted_cv}")
    #     stratified_cv = StratifiedKFold(n_splits=adjusted_cv, shuffle=True, random_state=42)
    # else:  # Class is large enough for normal CV
    #     stratified_cv = StratifiedKFold(n_splits=original_cv, shuffle=True, random_state=42)
    #     print(f"Using {original_cv} folds for grid search")
    
    original_cv_splits = n_splits
    if min_samples < original_cv_splits: 
        adjusted_cv_splits = max(2, min(3, min_samples)) 
        print(f"Adjusting CV splits: Minority class has only {min_samples} samples, "
              f"reducing folds from {original_cv_splits} to {adjusted_cv_splits}") 
        cv_strategy = StratifiedKFold(n_splits=adjusted_cv_splits, shuffle=True, random_state=random_state) 
    else: 
        cv_strategy = StratifiedKFold(n_splits=original_cv_splits, shuffle=True, random_state=random_state)
        print(f"Using {original_cv_splits} folds for CV")
    
 
    
    # Adjust n_splits if necessary
    # original_splits = n_splits
    # if min_samples < n_splits:
    #     # Use at most min_samples splits, but at least 2 if possible
    #     n_splits = max(2, min(min_samples, 3))
    #     print(f"\nAdjusting cross-validation: Minority class has only {min_samples} samples, "
    #           f"reducing folds from {original_splits} to {n_splits}")
    
    # print(f"\nEvaluating {model_name} with {n_splits}-fold cross-validation...")
    
    #---------------Which CV????? ------------------------------
    #---Option 1>>  Stratified K-Fold cross-validation
    # Define scoring metrics, added zero_division=0 for warning
    is_binary_cv = len(np.unique(y)) == 2
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
        'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
        'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0)
    }
    if is_binary_cv:
        scoring['roc_auc'] = make_scorer(roc_auc_score, needs_proba=True)
        scoring['average_precision'] = make_scorer(average_precision_score, needs_proba=True)
    else: # Multiclass
        scoring['roc_auc_ovr_macro'] = make_scorer(roc_auc_score, needs_proba=True, multi_class='ovr', average='macro')
        # For AP, 'weighted' average is generally preferred for multiclass consistency with F1
        scoring['average_precision_weighted'] = make_scorer(average_precision_score, needs_proba=True, average='weighted')
    

    
    # scoring = {
    #     'accuracy': make_scorer(accuracy_score),
    #     'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
    #     'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    #     'recall': make_scorer(recall_score, average='weighted', zero_division=0)
    # }

    # Limit parallel jobs to reduce memory usage
    # n_jobs = 1  # Avoid parallel issues with small datasets
    # print(f"\nUsing Stratified K-Fold cross-validation for {model_name}...")
    # cv = stratified_cv
    
    # cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    #  OR>>>>>
    # print(f"\nUsing repeated Stratified K-Fold cross-validation for {model_name}...")
    # cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5, random_state=42)
    
    # Perform cross-validation
    # cv_results = cross_validate(
    #     model, X, y,
    #     cv=cv,
    #     scoring=scoring,
    #     return_train_score=True,
    #     n_jobs=1)
    # Calculate and print results
    # metrics = ['accuracy', 'f1_weighted', 'precision', 'recall']
    # print(f"\nCross-validation results for {model_name}:")
    # print(f"{'Metric':<15} {'Train Mean':<10} {'Train Std':<10} {'Test Mean':<10} {'Test Std':<10} {'Gap':<10}")
    # print("-" * 70)
    
    
    n_jobs = 1 
    print(f"\nUsing Stratified K-Fold cross-validation for {model_name}...")
    cv_results_dict = cross_validate( 
        model, X, y,
        cv=cv_strategy, # Use the determined CV strategy
        scoring=scoring,
        return_train_score=True,
        n_jobs=n_jobs)
    metrics_to_report = list(scoring.keys()) # Get all keys from the scoring dict

    print(f"\nCross-validation results for {model_name}:")
    header = f"{'Metric':<30} {'Train Mean':<12} {'Train Std':<10} {'Test Mean':<12} {'Test Std':<10} {'Gap':<10}" 
    print(header) 
    print("-" * len(header))

    
    # for metric in metrics:
    #     train_scores = cv_results[f'train_{metric}']
    #     test_scores = cv_results[f'test_{metric}']
        
    #     train_mean = np.mean(train_scores)
    #     train_std = np.std(train_scores)
    #     test_mean = np.mean(test_scores)
    #     test_std = np.std(test_scores)
    #     gap = train_mean - test_mean
        
    #     print(f"{metric:<15} {train_mean:.4f} {train_std:.4f} {test_mean:.4f} {test_std:.4f} {gap:.4f}")
    
    for metric_key_name in metrics_to_report: # Iterate using the actual keys used in scoring
        train_scores = cv_results_dict.get(f'train_{metric_key_name}', np.array([np.nan])) # [cite: 664]
        test_scores = cv_results_dict.get(f'test_{metric_key_name}', np.array([np.nan])) # [cite: 664]

        train_mean = np.nanmean(train_scores) # [cite: 665]
        train_std = np.nanstd(train_scores) # [cite: 665]
        test_mean = np.nanmean(test_scores) # [cite: 665]
        test_std = np.nanstd(test_scores) # [cite: 665]
        gap = train_mean - test_mean if not (np.isnan(train_mean) or np.isnan(test_mean)) else np.nan # [cite: 665]

        print(f"{metric_key_name:<30} {train_mean:.4f} {train_std:.4f} {test_mean:.4f} {test_std:.4f} {gap:.4f}") # [cite: 665]

    
    # Check for overfitting
    # f1_gap = np.mean(cv_results['train_f1_weighted']) - np.mean(cv_results['test_f1_weighted'])
    # if f1_gap > 0.1:
    #     print(f"\nWarning: Possible overfitting detected (F1 gap: {f1_gap:.4f})")
    #     if f1_gap > 0.2:
    #         print("Severe overfitting detected!")
    
    # # Also evaluate on individual folds to check consistency
    # print("\nPerformance on individual folds:")
    # for i, (train_f1, test_f1) in enumerate(zip(cv_results['train_f1_weighted'], cv_results['test_f1_weighted'])):
    #     print(f"Fold {i+1}: Train F1 = {train_f1:.4f}, Test F1 = {test_f1:.4f}, Gap = {train_f1 - test_f1:.4f}")
    
    f1_gap = np.nanmean(cv_results_dict.get('train_f1_weighted', np.nan)) - np.nanmean(cv_results_dict.get('test_f1_weighted', np.nan)) 
    if not np.isnan(f1_gap):
        if f1_gap > 0.1: 
            print(f"\nWarning: Possible overfitting detected (F1 gap: {f1_gap:.4f})") 
            if f1_gap > 0.2:
                print("Severe overfitting detected!") 

    print("\nPerformance on individual folds (F1 Weighted):") 
    train_f1_folds = cv_results_dict.get('train_f1_weighted', []) 
    test_f1_folds = cv_results_dict.get('test_f1_weighted', []) 
    for i, (train_f1_fold, test_f1_fold) in enumerate(zip(train_f1_folds, test_f1_folds)):
        print(f"Fold {i+1}: Train F1 = {train_f1_fold:.4f}, Test F1 = {test_f1_fold:.4f}, Gap = {train_f1_fold - test_f1_fold:.4f}") 
    

    
    
    # #---Option 3>>   LeaveOneOut for final evaluation not recommended
    # issues: The zero_division=0 parameter tells sklearn to return 0 instead of raising a warning when division by zero occurs in precision or recall calculations. This happens frequently with LOOCV on small datasets because:
    # Each fold contains only one sample, When that sample is from a minority class, its entire class may be missing from the training set. The model might not predict certain classes at all in some folds
    
    # print(f"Evaluating {model_name} with Leave-One-Out CV ({len(y)} iterations)")
    
    # loocv = LeaveOneOut()
    # scoring = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']
    
    # cv_results = cross_validate(
    #     model, X, y,
    #     cv=loocv,
    #     scoring=scoring,
    #     return_train_score=False,  # Train scores not meaningful with LOOCV
    #     n_jobs=1  # Single job to avoid memory issues
    # )
    
    # for metric in scoring:
    #     metric_name = metric.replace('_weighted', '')
    #     scores = cv_results[f'test_{metric}']
    #     print(f"{metric_name}: {scores.mean():.4f} (+/- {scores.std():.4f})")


    
    
    return cv_results_dict # cv_results



def run_classification_pipeline(data_path, normalization='standard', sampling_method='none', 
                              is_binary=True, preserve_zones=False, sort_features='none',
                              feature_indices=None, selected_features=None, tune_hyperparams=False,
                              analyze_shap=False, analyze_minkowski_dist=False,
                              transform_features=False, report_dir=None): 
    set_seeds() 
    print(f"Loading data from {data_path}...") 
    data = pd.read_csv(data_path) 

    if selected_features is not None: 
        data = data[selected_features + [data.columns[-1]]] 
    # feature_indices handled by prepare_data

    print(f"Preparing data (binary={is_binary}, preserve_zones={preserve_zones}, sort_features={sort_features}, normalization={normalization}, transform_features={transform_features})...") 
    initial_num_features_run = len(feature_indices) if feature_indices is not None else 9

    X, y, le, scaler, feature_names, actual_group_metrics = prepare_data(
        data, 
        num_features=initial_num_features_run, 
        values_per_feature=20, # Default, or make parameter
        normalization=normalization, 
        is_binary=is_binary, 
        preserve_zones=preserve_zones, 
        sort_features=sort_features, 
        feature_indices=feature_indices,
        transform_features=transform_features 
    )
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y) 
    y_train_original = y_train.copy() 
    print("Printing split statistics to verify stratified balance \n")
    print(f"Training set class distribution: {Counter(y_train)}") 
    print(f"Test set class distribution: {Counter(y_test)}") 

    if np.isnan(X_train).any(): 
        print(f"Warning: {np.isnan(X_train).sum()} NaN values detected in training data.") 
        from sklearn.impute import SimpleImputer 
        imputer = SimpleImputer(strategy='mean') # 
        X_train = imputer.fit_transform(X_train); X_test = imputer.transform(X_test) 
    
    from .reporting.utils import save_figure 
    # Global sampling logic
    if not tune_hyperparams and sampling_method != 'none':  
        print(f"Applying {sampling_method} sampling globally...")
        if sampling_method == 'smote': 
            min_class_samples = min(np.bincount(y_train)[np.bincount(y_train) > 0]) # [cite: 798]
            k_neighbors = min(min_class_samples - 1, 3); # [cite: 798]
            sampler = SMOTE(random_state=42, k_neighbors=k_neighbors) if k_neighbors >=1 else RandomOverSampler(random_state=42) # [cite: 798, 799]
        elif sampling_method == 'random_over': sampler = RandomOverSampler(random_state=42) 
        elif sampling_method == 'random_under': sampler = RandomUnderSampler(random_state=42) 
        else: sampler = None
        if sampler: X_train, y_train = sampler.fit_resample(X_train, y_train) 
        print(f"Applied {sampling_method}. New training set shape: {X_train.shape}, New label distribution: {Counter(y_train)}") 
        # Plotting distributions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.bar(np.unique(y_train_original), np.bincount(y_train_original)); ax1.set_title('Original'); 
        ax2.bar(np.unique(y_train), np.bincount(y_train)); ax2.set_title(f'After {sampling_method.upper()}'); # [cite: 806, 807]
        plt.tight_layout(); save_figure(plt.gcf(), f"1_class_distribution_comparison_{sampling_method}_{normalization}", report_dir); plt.close(); # [cite: 807]

    elif sampling_method == 'none': # [cite: 808]
        print("No global sampling method selected.") # [cite: 808]
        plot_class_distribution(y_train_original, le, '1_class_distribution_Original') # [cite: 808]
        plt.tight_layout(); save_figure(plt.gcf(), f"1_class_distribution_Original_{sampling_method}_{normalization}", report_dir); plt.close() # [cite: 808, 809]
    else: # tune_hyperparams is True and sampling_method is not 'none'
         print(f"Skipping global sampling (tune_hyperparams=True). Sampling ('{sampling_method}') will be handled within CV if applicable.") # [cite: 807, 808]
         plot_class_distribution(y_train_original, le, '1_class_distribution_Original_CV_Sampling') #
         plt.tight_layout(); save_figure(plt.gcf(), f"1_class_distribution_Original_CV_Sampling_{sampling_method}_{normalization}", report_dir); plt.close() #


    if analyze_minkowski_dist: analyze_minkowski(X_train, p_values=[1, 2, 3])  
    num_classes = len(np.unique(y)) 
    class_distribution = np.bincount(y_train) 
    base_models = create_base_models(num_classes, class_distribution) 
    
    if tune_hyperparams: # [cite: 810]
        print(f"Tuning hyperparameters for {'binary' if is_binary else 'multiclass'} classification...") # [cite: 810]
        param_grids = binary_param_grids if is_binary else multiclass_param_grids # [cite: 810, 811]
        tuned_models = {} # [cite: 813]
        for name, model_instance in base_models.items(): # [cite: 813]
            if name in param_grids: # [cite: 813]
                model_grid = param_grids[name] # [cite: 814]
                # Pass sampling_method to grid_search_model [cite: 815]
                best_model_tuned, best_params = grid_search_model(model_instance, model_grid, X_train, y_train, name, cv=5, report_dir=report_dir, sampling_method=sampling_method) # [cite: 814, 815]
                tuned_models[name] = best_model_tuned # [cite: 815]
                print(f"Best parameters for {name}: {best_params}") # [cite: 816]
            else:
                tuned_models[name] = model_instance # [cite: 816]
        base_models = tuned_models # [cite: 817]

    results = {}; accuracies = {}; f1_scores_map = {}; trained_models = {} 
    for name, model_to_train in base_models.items(): 
        trained_model_instance, y_pred, roc_auc, avg_precision = train_evaluate_model(model_to_train, X_train, X_test, y_train, y_test, name, le) 
        trained_models[name] = trained_model_instance 
        acc = accuracy_score(y_test, y_pred); f1 = f1_score(y_test, y_pred, average='weighted') 
        accuracies[name] = acc; f1_scores_map[name] = f1 
        save_results(name, acc, roc_auc, avg_precision, str(trained_model_instance.get_params()), X.shape[1], data_path, is_binary, preserve_zones, sort_features, normalization, sampling_method, y_test, y_pred) 
    
    print("\nModel Performance Comparison:") 
    for name in accuracies: print(f"{name} - Accuracy: {accuracies[name]:.4f}, F1 Score: {f1_scores_map[name]:.4f}") 
    
    best_model_name = max(f1_scores_map, key=f1_scores_map.get) if f1_scores_map else None 
    best_model = trained_models.get(best_model_name) if best_model_name else None

    if best_model_name and best_model: 
        print(f"\nBest model: {best_model_name} (F1 Score: {f1_scores_map[best_model_name]:.4f})") # [cite: 860]
        # ... (plot_learning_curve, save_best_model as before) ... 
        if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Logistic Regression', 'SVM']: # [cite: 860]
             plot_learning_curve(best_model, X_train, y_train, X_test, y_test) 
        save_best_model(best_model, best_model_name, { # [cite: 861]
            'is_binary': is_binary, 'preserve_zones': preserve_zones, 'sort_features': sort_features, # [cite: 861]
            'normalization': normalization, 'sampling_method': sampling_method, 'num_features': X.shape[1], # [cite: 861]
            'accuracy': accuracies[best_model_name], 'f1_score': f1_scores_map[best_model_name] # [cite: 861]
        })


    if analyze_shap and best_model is not None: 
        print(f"\nPerforming SHAP analysis on the best model ({best_model_name})...") 
        # feature_names and actual_group_metrics are available from prepare_data
        common_shap_args_run = {
            "X_test": X_test, "feature_names": feature_names, "report_dir": report_dir,
            "is_binary": is_binary, "y_test": y_test,
            "group_metrics": actual_group_metrics, 
            "num_feature_groups": len(actual_group_metrics),
            "values_per_group": 20 # Default, or pass from config
        }
        try: 
            run_shap_analysis(best_model, plot_type='summary', **common_shap_args_run) 
            run_shap_analysis(best_model, plot_type='bar', **common_shap_args_run)
            if isinstance(best_model, (RandomForestClassifier, GradientBoostingClassifier, xgb.XGBClassifier)): 
                run_shap_analysis(best_model, plot_type='beeswarm', **common_shap_args_run) 
            sample_idx = 0 
            run_shap_analysis(best_model, sample_idx=sample_idx, plot_type='waterfall', **common_shap_args_run) # [cite: 881]
            run_shap_analysis(best_model, sample_idx=sample_idx, plot_type='force', **common_shap_args_run) # [cite: 881]
            if len(X_test) > 1: run_shap_analysis(best_model, sample_idx=1, plot_type='waterfall', **common_shap_args_run) 
            if is_binary and len(X_test) > 5: 
                minority_idx = np.where(y_test == np.bincount(y_test).argmin())[0] 
                if len(minority_idx) > 0: 
                    sample_idx = np.random.choice(minority_idx) 
                    run_shap_analysis(best_model, sample_idx=sample_idx, plot_type='waterfall', **common_shap_args_run) 
        except Exception as e: 
            print(f"Error during SHAP analysis calls: {e}"); traceback.print_exc() 

        # Call analyze_feature_group_importance with actual metrics
        if best_model:
            num_classes_for_loop = len(le.classes_) if not is_binary else 1 #
            for c_idx in range(num_classes_for_loop): #
                class_to_analyze = c_idx if not is_binary else 0 # For binary, SHAP often gives for class 1, but importance is general or for positive
                print(f"Analyzing feature group importance for class {class_to_analyze if not is_binary else 'positive'}")
                analyze_feature_group_importance(
                    best_model, X_test,
                    num_features=len(actual_group_metrics),
                    values_per_feature=20, # Or from config
                    metrics=actual_group_metrics,
                    class_index=class_to_analyze, # Pass the class index
                    report_dir=report_dir,
                    y_test=y_test
                )
            # The multi-class plotting for analyze_feature_group_importance was removed as it's better to call it per class.
            # The old logic for subplots is replaced by the loop above.


    cv_results_data_run = None
    if best_model:
        cv_results_data_run = evaluate_model_with_cv(best_model, X, y, best_model_name, n_splits=10, random_state=42) # [cite: 905, 906]
    
    best_model_f1_run = f1_scores_map.get(best_model_name, 0.0) if best_model_name else 0.0 # [cite: 907, 908]

    return best_model, { # [cite: 908]
        'trained_models': trained_models, 'accuracies': accuracies, 'f1_scores': f1_scores_map, # [cite: 908]
        'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test, 'le': le, # [cite: 908, 909]
        'feature_names': feature_names, 'best_model_name': best_model_name, # [cite: 909]
        'f1_score': best_model_f1_run, # [cite: 909]
        'actual_group_metrics': actual_group_metrics,
        'cv_results': cv_results_data_run # [cite: 909]
    }


def run_classification_pipeline_og(data_path, normalization='standard', sampling_method='none', 
                              is_binary=True, preserve_zones=False, sort_features='none',
                              feature_indices=None, selected_features=None, tune_hyperparams=False,
                              analyze_shap=False, analyze_minkowski_dist=False,
                              transform_features=False, report_dir=None):
    """
    Run classification pipeline with all options
    
    Args:
        data_path: Path to data
        normalization: Normalization technique
        sampling_method: Sampling method for class imbalance
        is_binary: Whether to use binary classification
        preserve_zones: Whether to preserve zone-wise structure
        sort_features: Sorting strategy ('none', 'ascend_all', 'descend_all', 'custom')
        feature_indices: List of specific feature indices to use (0-8)
        selected_features: List of selected features
        tune_hyperparams: Whether to tune hyperparameters
        analyze_shap: Whether to perform SHAP analysis
        analyze_minkowski_dist: Whether to analyze Minkowski distances
        report_dir: Directory to save reports and visualizations
        
    Returns:
        best_model: Best model
        results: Dictionary of results
    """
    # Set seeds for reproducibility
    set_seeds()
    
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
    # No else clause needed - use all features by default
        
    
    print(f"Preparing data (binary={is_binary}, preserve_zones={preserve_zones}, sort_features={sort_features}, normalization={normalization}, transform_features=False)...")
    X, y, le, scaler, feature_names = prepare_data(data, 
                                        normalization=normalization, 
                                        is_binary=is_binary,
                                        preserve_zones=preserve_zones,
                                        sort_features=sort_features,
                                        transform_features=False)  # No feature transformation yet
    
    # Split data BEFORE any advanced feature transformations
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    y_train_original = y_train.copy()
    print("Printing split statistics to verify stratified balance \n")
    print(f"Training set class distribution: {Counter(y_train)}")
    print(f"Test set class distribution: {Counter(y_test)}")
    
    # Now apply feature transformation separately to train and test sets
    # Important: fit only on training data, transform both
    if transform_features:
        print("Applying feature transformations...")
        # Calculate means, stds, etc. for features using ONLY training data
        X_train_transformed, train_feature_names = create_feature_transformations_transform(
            X_train, feature_names, 
            num_features=len(feature_indices) if feature_indices is not None else 9,
            values_per_feature=20
        )
        
        # For test data, we need to implement a transformation that uses
        # the same statistics as computed from the training data
        # This requires modifying create_feature_transformations to save and reuse stats
        # For now, we'll just transform test data independently (not ideal)
        X_test_transformed = create_feature_transformations_transform(
            X_test, params, feature_names=feature_names,
            num_features=len(feature_indices) if feature_indices is not None else 9,
            values_per_feature=20
        )
        
        # Update X_train and X_test
        X_train = X_train_transformed
        X_test = X_test_transformed
        feature_names = train_feature_names
        
    
    # Check for NaN values before applying sampling
    if np.isnan(X_train).any():
        print(f"Warning: {np.isnan(X_train).sum()} NaN values detected in training data.")
        print("Replacing NaN values with feature means...")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)  # Use same imputer for test data
        
        # Verify NaNs are gone
        if np.isnan(X_train).any() or np.isnan(X_test).any():
            print("Error: NaN values still present after imputation.")
            raise ValueError("Failed to remove all NaN values")
        else:
            print("NaN values successfully removed.")
    
    # Plot original class distribution
    # plot_class_distribution(y_train, le, 'Original Class Distribution')
    from report_generation import save_figure
    
    # Option 1: Choose sampling methods (Globally)- ONLY to training data
    if not tune_hyperparams and sampling_method != 'none': 
        print(f"Applying {sampling_method} sampling to training data...")
        sampler = None # Initialize
        if sampling_method == 'smote':
            # Calculate minimum safe k value (neighbors) for SMOTE
            min_class_samples = min(np.bincount(y_train)[np.bincount(y_train) > 0])
            k_neighbors = min(min_class_samples - 1, 3)  # At least 1, at most 5
            if k_neighbors < 1:
                print("Warning: Smallest class has too few samples for SMOTE. Using RandomOverSampler instead.")
                sampler = RandomOverSampler(random_state=42)
            else:
                print(f"Using SMOTE with k_neighbors={k_neighbors} for small classes.")
                sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
        elif sampling_method == 'random_over':
            sampler = RandomOverSampler(random_state=42)
        elif sampling_method == 'random_under':
            sampler = RandomUnderSampler(random_state=42)
        else: print(f"Warning: Unknown global sampling_method '{sampling_method}'. No sampling applied.")

        if sampler: # Proceed only if a sampler was initialized
            X_train, y_train = sampler.fit_resample(X_train, y_train)
            print(f"Applied {sampling_method}. New training set shape: {X_train.shape}, New label distribution: {Counter(y_train)}")
    #  Option 2: More nuanced balancing
    # if sampling_method != 'none':
    #     print(f"Applying {sampling_method} sampling...")
        
    #     # Special handling for very small datasets
    #     if len(y_train) < 100:
    #         # For tiny datasets, use synthetic minority oversampling but with careful parameters
    #         from imblearn.over_sampling import SMOTENC, SMOTE
            
    #         # Check if any categorical features present
    #         categorical_mask = [False] * X_train.shape[1]  # Assume all numeric
            
    #         # Calculate k neighbors parameter based on minority class size
    #         class_counts = np.bincount(y_train)
    #         min_samples = min(class_counts[class_counts > 0])
    #         k_neighbors = min(min_samples-1, 3)  # Use at most 3 neighbors, must be less than min class size
            
    #         if k_neighbors < 1:
    #             print("Warning: Minority class too small for SMOTE. Using random oversampling.")
    #             from imblearn.over_sampling import RandomOverSampler
    #             sampler = RandomOverSampler(random_state=42)
    #         else:
    #             print(f"Using SMOTE with k_neighbors={k_neighbors} for small dataset")
    #             # If continuous data only
    #             sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
            
    #         X_train, y_train = sampler.fit_resample(X_train, y_train)
    #         print(f"After balancing: {np.bincount(y_train)}") 
        
        # plot_class_distribution(y_train, le, f'Class Distribution After {sampling_method}')
    
        # create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot original distribution
        ax1.bar(np.unique(y_train_original), np.bincount(y_train_original))
        ax1.set_title('Original Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        
        # Plot distribution after sampling
        ax2.bar(np.unique(y_train), np.bincount(y_train))
        ax2.set_title(f'After {sampling_method.upper()} ({normalization.capitalize()})')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Count')
        plt.tight_layout()
        if report_dir: save_figure(plt.gcf(), f"1_class_distribution_comparison_{sampling_method}_{normalization}", report_dir)
        plt.close(fig)
    else:
        if sampling_method != 'none' and tune_hyperparams:
            print(f"Skipping global sampling (tune_hyperparams=True). Sampling will be handled within CV if applicable.")
        elif sampling_method == 'none': 
            print("No sampling method selected.")
            # figsize=(16, 6)
            # plt.bar(np.unique(y_train_original), np.bincount(y_train_original))
        fig = plt.figure(figsize=(8,6))
        plt.xlabel('Class')
        plt.ylabel('Count')
        plot_class_distribution(y_train_original, le, '1_class_distribution_Original')
        plt.tight_layout()
        if report_dir: save_figure(plt.gcf(), f"1_class_distribution_Original_{sampling_method}_{normalization}", report_dir)
        plt.close(fig)

    
    # Analyze Minkowski distances if requested
    if analyze_minkowski_dist:
        print("Analyzing Minkowski distances...")
        analyze_minkowski(X_train, p_values=[1, 2, 3])
    
    # Define models
    num_classes = len(np.unique(y))
    print(f"Number of classes: {num_classes}")
    class_distribution = np.bincount(y_train)
    print(f"Class distribution in training data: {class_distribution}")
    

    # Define models with optimized configurations
    base_models = create_base_models(num_classes, class_distribution)
    
    
    # Tune hyperparameters if requested
    if tune_hyperparams:
        print(f"Tuning hyperparameters for {'binary' if is_binary else 'multiclass'} classification...")
        
        # Choose appropriate parameter grids based on classification type
        if is_binary:
            print("Using specialized parameter grids for binary classification")
            param_grids = binary_param_grids
        else:
            print("Using parameter grids for multiclass classification")
            param_grids = multiclass_param_grids
        
        # Debug output - print all model names and their param grid sizes
        for name in base_models:
            if name in param_grids:
                grid_size = 1
                for param, values in param_grids[name].items():
                    grid_size *= len(values)
                print(f"Model {name}: Parameter grid has {grid_size} combinations")
        
        # Tune each model with appropriate grid
        tuned_models = {}
        for name, model in base_models.items():
            if name in param_grids:
                print(f"\n{'='*50}\nTuning hyperparameters for {name}\n{'='*50}")
                
                # Select parameter grid for this model
                model_grid = param_grids[name]
                
                # Run grid search to find best params
                best_model, best_params = grid_search_model(
                    model, model_grid, X_train, y_train, name, cv=3, report_dir=report_dir,sampling_method=sampling_method  # <-- ADDED
                )
                
                # Store tuned model
                tuned_models[name] = best_model
                
                print(f"Best parameters for {name}: {best_params}")
            else:
                # If no grid defined, keep original model
                print(f"No parameter grid defined for {name}, using default configuration")
                tuned_models[name] = model
        
        # Replace base models with tuned versions
        base_models = tuned_models
        
    #     # Define parameter grids for each model
    #     if is_binary:
    #         print("Using specialized parameter grids for binary classification")
    #         param_grids = {
    #             'XGBoost': {
    #                 'n_estimators': [100, 200],
    #                 'learning_rate': [0.01, 0.05],
    #                 'max_depth': [2, 3],
    #                 'subsample': [0.7, 0.8],
    #                 'colsample_bytree': [0.7, 0.8],
    #                 'scale_pos_weight': [1.0, sum(y_train==0)/sum(y_train==1)],  # Balanced weight
    #                 'min_child_weight': [1, 3],
    #                 'gamma': [0, 0.1]
    #             },
    #             # Add specialized grids for other models
    #         }
    #     else:
    #         # Original parameter grids
    #         # Define parameter grids for each model
    #         param_grids = {
    #         'Logistic Regression': {
    #             'C': [0.01, 0.05, 0.1],  # Focus on stronger regularization
    #             'penalty': ['l2'],  # Start with L2 only
    #             'solver': ['liblinear' if num_classes == 2 else 'saga']
    #         },
    #         'SVM': {
    #             'C': [0.01, 0.1, 1.0],
    #             'kernel': ['linear', 'rbf'],
    #             'gamma': ['scale', 'auto']
    #         },           
    #         'SVM_pipeline': {
    #             'poly__degree': [1, 2],  # Lower degree polynomials
    #             'svm__C': [0.1, 1.0],
    #             'svm__gamma': ['scale']
    #         },
    #         'Random Forest': {
    #             'n_estimators': [50, 100],
    #             'max_depth': [2, 3, 5],
    #             # 'min_samples_split': [5, 10],
    #             'min_samples_leaf': [4, 8, 10]
    #         },
    #         'Gradient Boosting': {
    #             'n_estimators': [50, 100],
    #             'learning_rate': [0.01, 0.05],
    #             'max_depth': [2, 3],
    #             'subsample': [0.7, 0.8],
    #             'min_samples_split': [5, 10]
    #         },
    #         'XGBoost': {
    #         #     'n_estimators': [50, 100],
    #         #     'learning_rate': [0.01, 0.05],
    #         #     'max_depth': [2, 3],
    #         #     'subsample': [0.7, 0.8],
    #         #     'colsample_bytree': [0.7, 0.8],
    #         #     'reg_alpha': [1, 2],
    #         #     'reg_lambda': [5],
    #         #     'min_child_weight': [3]
    #         # },
    #             'n_estimators': [30, 50],  # Fewer trees
    #             'learning_rate': [0.0005, 0.001, 0.05],  # Extremely slow learning
    #             'max_depth': [1,2],  # Force decision stumps (depth-1 trees)
    #             'min_child_weight': [3, 10, 15],  # 5,8,10 Require much more data per node
    #             'gamma': [10, 20],  # Very high minimum gain for splitting
    #             'subsample': [0.4, 0.5, 0.7],  # Sample only 40-50% of data per tree
    #             'colsample_bytree': [0.3, 0.8],  # Sample only 30-40% of features per tree
    #             'reg_alpha': [1,50, 100],  # Extreme L1 regularization
    #             'reg_lambda': [5,50, 100],  # Extreme L2 regularization
    #             # 'scale_pos_weight': [1],  # For multiclass, keep as 1
    #             # 'num_class': [4],  # Explicitly specify number of classes
    #             # 'objective': ['multi:softmax']
    #         },
    #         # Grid search on this pipeline
    #         'xgb_pipeline': {
    #             'feature_select__k': [5, 8, 10],
    #             'pca__n_components': [3, 4, 5],
    #             'xgb__learning_rate': [0.0005, 0.001, 0.005],
    #             'xgb__min_child_weight': [10, 15, 20]
    #         },
            
    #         'Naive Bayes': {'var_smoothing': [1e-7, 1e-6, 1e-5, 1e-4]
    #                         },  # Default> 1e-9}
    #         # 'Complement NB': {'alpha': [0.5, 1.0, 2.0], 
    #         #                 'norm': [True, False]
    #         #                 },
    #         # 'Multinomial NB': {
    #         #                     'alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
    #         #                     # 'fit_prior': [True, False]
    #         #                 },
    #         'Bernoulli NB': {
    #                         'alpha': [0.1, 2.0, 5.0, 10.0], 
    #                         'binarize': [0.0,0.05, 0.1],
    #                         'fit_prior': [True]  # Try with/without prior fitting
    #                         },
    #         'knn_model' : {
    #                         'n_neighbors': [5, 7, 9, 11],
    #                         'weights': ['uniform', 'distance'],
    #                         'metric': ['euclidean', 'cosine']
    #         },
    #         'stump'     : {
    #                         'min_samples_leaf': [5, 10, 15, 20],
    #                         'criterion': ['gini', 'entropy']
    #                     }        
    #     }

    #     # Add separate grid for elasticnet penalty (Logistic Regression)
    #         if 'Logistic Regression' in base_models and num_classes > 2:
    #             elasticnet_grid = {
    #                 'C': [0.01, 0.05, 0.1],
    #                 'penalty': ['elasticnet'],
    #                 'solver': ['saga'],  # Only saga supports elasticnet
    #                 'l1_ratio': [0.2, 0.5, 0.8]
    #             }
                
    #             # Try both L2 and elasticnet separately
    #             l2_model, l2_params = grid_search_model(
    #                 LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42),
    #                 param_grids['Logistic Regression'], X_train, y_train, "Logistic Regression (L2)"
    #             )
                
    #             elasticnet_model, elasticnet_params = grid_search_model(
    #                 LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42),
    #                 elasticnet_grid, X_train, y_train, "Logistic Regression (Elasticnet)"
    #             )
                
    #             # Compare models on validation set
    #             X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    #                 X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    #             )
                
    #             l2_model.fit(X_train_sub, y_train_sub)
    #             elasticnet_model.fit(X_train_sub, y_train_sub)
                
    #             l2_score = f1_score(y_val, l2_model.predict(X_val), average='weighted')
    #             elasticnet_score = f1_score(y_val, elasticnet_model.predict(X_val), average='weighted')
                
    #             # Choose the better model
    #             if elasticnet_score > l2_score:
    #                 base_models['Logistic Regression'] = elasticnet_model
    #                 print(f"Using elasticnet model (F1: {elasticnet_score:.4f})")
    #             else:
    #                 base_models['Logistic Regression'] = l2_model
    #                 print(f"Using L2 model (F1: {l2_score:.4f})")
                
    #             # Skip regular grid search for Logistic Regression
    #             del param_grids['Logistic Regression']
    #             # del base_models['Logistic Regression']    
    #             # Tune hyperparameters for each model
    #             for name in base_models:
    #                 if name in param_grids:
    #                     base_models[name], best_params = grid_search_model(
    #                         base_models[name], param_grids[name], X_train, y_train, name
    #                     )
    #                     print(f"Best parameters for {name}: {best_params}")
    #                 else:
    #                     print(f"Skipping grid search for {name} (no parameter grid defined)")
    
    # # # Create a voting classifier with all models
    # # voting_clf = VotingClassifier(
    # #     estimators=[(name, model) for name, model in base_models.items()],
    # #     voting='soft' if all(hasattr(model, 'predict_proba') for model in base_models.values()) else 'hard'
    # # )
    # # voting classifier with only 3 diverse models
    # voting_clf = VotingClassifier(
    # estimators=[
    #     ('lr', base_models['Logistic Regression']),
    #     ('rf', base_models['Random Forest']),
    #     ('svm', base_models['SVM'])
    #     ],
    #     voting='soft'
    # )
    # base_models['Voting Classifier'] = voting_clf
    
    # # Create a stacking classifier
    # base_classifiers = {
    #     'rf': RandomForestClassifier(random_state=42),
    #     'gb': GradientBoostingClassifier(random_state=42),
    #     'lr': LogisticRegression(random_state=42, max_iter=5000,  solver='liblinear' if num_classes == 2 else 'saga')
    # }

    # # Create a stacking classifier with hyperparameters
    # stacking_params_svc = {
    #     'stacking_classifier__rf__max_depth': [2, 3, 5],
    #     'stacking_classifier__rf__min_samples_leaf': [5, 10],
    #     'stacking_classifier__gb__learning_rate': [0.01, 0.05],
    #     'stacking_classifier__gb__max_depth': [2, 3],
    #     'stacking_classifier__lr__C': [0.1, 1.0, 10.0],
    #     'stacking_classifier__final_estimator__C': [0.1, 1.0, 10.0],
    #     'stacking_classifier__final_estimator__kernel': ['linear', 'rbf'],
    #     'stacking_classifier__final_estimator__degree': [2, 3],
    #     # 'stacker__final_estimator__C': [0.1, 1.0, 10.0],
    #     # 'stacker__final_estimator__kernel': ['linear', 'rbf'], # Removed 'poly' for simplicity
    #     'stacking_classifier__final_estimator__gamma': ['scale', 0.1, 1.0], # Only for RBF
    #     'stacking_classifier__final_estimator__class_weight': [None, 'balanced']
    # }
    
    # stacking_params_lr = {
    #     'stacking_classifier__rf__max_depth': [2, 3, 5],
    #     'stacking_classifier__rf__min_samples_leaf': [5, 10],
    #     'stacking_classifier__gb__learning_rate': [0.01, 0.05],
    #     'stacking_classifier__gb__max_depth': [2, 3],
    #     'stacking_classifier__lr__C': [0.1, 1.0, 10.0],
    #     'stacking_classifier__final_estimator__C': [0.001, 0.1, 1.0, 10.0],
    #     # 'stacking_classifier__final_estimator__kernel': ['poly', 'rbf'],
    #     # 'stacking_classifier__final_estimator__degree': [2, 3],
    #     'stacking_classifier__final_estimator__penalty': ['l2'], 
    #     'stacking_classifier__final_estimator__solver': ['liblinear'] ,
    #     'stacking_classifier__final_estimator__class_weight': [None, 'balanced'],  
    # }

    # # Add stacking classifier to base_models
    # base_models['Stacking Classifier'] = StackingClassifier(
    #     estimators=[(name, clf) for name, clf in base_classifiers.items()],
    #     final_estimator=SVC(probability=True, random_state=42),
    #     cv=5
    # )
    # # Add stacking params to param_grids if hyperparameter tuning is enabled
    # if tune_hyperparams:
    #     param_grids['Stacking Classifier'] = stacking_params_svc
    #     # param_grids['Stacking Classifier'] = stacking_params_lr
    #     # base_models['Stacking Classifier'] = StackingClassifier(
    #     #     estimators=[(name, clf) for name, clf in base_classifiers.items()],
    #     #     final_estimator=LogisticRegression(random_state=42, max_iter=3000, solver='liblinear'),
    #     #     cv=5
    #     # )
    
    # Train and evaluate models
    print("Training and evaluating models...")
    results = {}
    accuracies = {}
    f1_scores = {}
    roc_auc_scores = {} # added
    avg_precision_scores = {} 
    trained_models = {}
    
    # for name, model in base_models.items():
    #     trained_model, y_pred = train_evaluate_model(
    #         model, X_train, X_test, y_train, y_test, name, le
    #     )
    #     trained_models[name] = trained_model
    #     accuracies[name] = accuracy_score(y_test, y_pred)
    #     f1_scores[name] = f1_score(y_test, y_pred, average='weighted')
        
    #     # Save results
    #     save_results(
    #         name, accuracies[name], str(trained_model.get_params()),
    #         X.shape[1], data_path, is_binary, preserve_zones, sort_features,
    #         normalization, sampling_method, y_test, y_pred
    #     )
    for name, model_instance in base_models.items(): # Changed 'model'->'model_instance' to avoid conflict
        trained_model_instance, y_pred, roc_auc, avg_precision = train_evaluate_model( # Capture new metrics
            model_instance, X_train, X_test, y_train, y_test, name, le 
        )
        trained_models[name] = trained_model_instance 
        accuracies[name] = accuracy_score(y_test, y_pred) 
        current_f1 = f1_score(y_test, y_pred, average='weighted') 
        f1_scores[name] = current_f1 
        roc_auc_scores[name] = roc_auc
        avg_precision_scores[name] = avg_precision

        save_results( 
            name, accuracies[name], roc_auc_scores[name], avg_precision_scores[name],
            str(trained_model_instance.get_params() if hasattr(trained_model_instance, 'get_params') else {}),
            X.shape[1], data_path, is_binary, preserve_zones, sort_features, 
            normalization, sampling_method, y_test, y_pred 
        )
    
    # Print performance comparison
    print("\nModel Performance Comparison:")
    for name in accuracies:
        print(f"{name} - Accuracy: {accuracies[name]:.4f}, F1 Score: {f1_scores[name]:.4f}, ROC AUC: {roc_auc_scores.get(name, 'N/A')}, Avg Precision: {avg_precision_scores.get(name, 'N/A')}") 
    
    best_model_name = None
    best_model_instance_overall = None
    
    # Find best model based on F1 score
    # best_model_name = max(f1_scores, key=f1_scores.get)
    # best_model = trained_models[best_model_name]
    # best_model_f1_score = f1_scores[best_model_name]
    if f1_scores:
        best_model_name = max(f1_scores, key=f1_scores.get) #if f1_scores else None 
        best_model_instance_overall = trained_models.get(best_model_name) #if best_model_name else None

    
    # print(f"\nBest model: {best_model_name} (F1 Score: {f1_scores[best_model_name]:.4f})")
    # # Plot learning curve for the best model
    # if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Logistic Regression', 'SVM']:
    #     print(f"\nPlotting learning curve for {best_model_name}...")
    #     plot_learning_curve(best_model, X_train, y_train, X_test, y_test) 
    
    if best_model_instance_overall and best_model_name:  
        print(f"\nBest model: {best_model_name} (F1 Score: {f1_scores[best_model_name]:.4f})") 
        if best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'Logistic Regression', 'SVM']: 
            print(f"\nPlotting learning curve for {best_model_name}...") 
            if report_dir: 
                plot_learning_curve(best_model_instance_overall, X_train, y_train, X_test, y_test, cv=3, n_jobs=1) 
           
        save_best_model(best_model_instance_overall, best_model_name, {
        'is_binary': is_binary,
        'preserve_zones': preserve_zones,
        'sort_features': sort_features,
        'normalization': normalization,
        'sampling_method': sampling_method,
        'num_features': X.shape[1],
        'accuracy': accuracies.get(best_model_name, 0.0), # Use .get for safety
        'f1_score': f1_scores.get(best_model_name, 0.0),
        # 'selection_criteria': 'max_f1_test_set',
        'roc_auc': roc_auc_scores.get(best_model_name, 0.0), # Added
        'avg_precision': avg_precision_scores.get(best_model_name, 0.0)
        })
    else:
        print("\nNo models were successfully trained to determine the best model.")

    # --- START: NEW LOGIC for Top 3 Gap+F1 Models ---
    # print(f"\n{'='*30} Selecting Top 3 Models (Low Gap + High F1) {'='*30}")

    # final_model_evals = {}
    # # Evaluate all trained models using final CV on the full dataset (X, y)
    # # This ensures consistent comparison based on CV performance and gap
    # for name, model in trained_models.items():
    #     print(f"Performing final CV evaluation for {name}...")
    #     try:
    #         # Ensure X, y are the versions AFTER initial preparation but BEFORE sampling/transformation specific to training loop if tune_hyperparams=False
    #         # If tune_hyperparams=True, the CV within grid_search already did this, but re-evaluating ensures consistency
    #         final_cv_results = evaluate_model_with_cv(
    #             model, X, y, name, n_splits=5, random_state=42 # Using 5 splits for consistency
    #         )

    #         test_f1_mean = np.mean(final_cv_results['test_f1_weighted'])
    #         train_f1_mean = np.mean(final_cv_results['train_f1_weighted'])
    #         # Handle potential NaN if a metric calculation failed in CV
    #         if np.isnan(train_f1_mean) or np.isnan(test_f1_mean):
    #             print(f"  Warning: NaN scores encountered during CV for {name}. Skipping this model for gap analysis.")
    #             continue # Skip if scores are NaN

    #         f1_gap = abs(train_f1_mean - test_f1_mean) # Use absolute gap

    #         final_model_evals[name] = {
    #             'f1_score': test_f1_mean,
    #             'gap': f1_gap,
    #             'model': model,
    #             # Store other metrics if needed
    #             'accuracy': np.mean(final_cv_results['test_accuracy'])
    #         }
    #         print(f"  {name}: Final CV Test F1={test_f1_mean:.4f}, Gap={f1_gap:.4f}")
    #     except Exception as cv_err:
    #         print(f"  Error during final CV for {name}: {cv_err}. Skipping.")
    #         traceback.print_exc()


    # if not final_model_evals:
    #     print("Could not perform final CV evaluations. Skipping Gap+F1 model saving.")
    # else:
    #     # Sort models: first by ascending gap, then by descending F1 score
    #     sorted_models = sorted(
    #         final_model_evals.items(),
    #         key=lambda item: (item[1]['gap'], -item[1]['f1_score']) # Sort by gap (asc), then F1 (desc)
    #     )

    #     print("\nModels ranked by Gap (asc) then F1 (desc):")
    #     for rank, (name, stats) in enumerate(sorted_models):
    #         print(f"  Rank {rank+1}: {name} (Gap: {stats['gap']:.4f}, F1: {stats['f1_score']:.4f})")

    #     # Select top 3
    #     top_3_models = sorted_models[:3]

    #     print(f"\nSaving Top {len(top_3_models)} Models based on Gap+F1...")
    #     for rank, (name, stats) in enumerate(top_3_models):
    #         top_model = stats['model']
    #         top_f1 = stats['f1_score']
    #         top_gap = stats['gap']
    #         top_acc = stats.get('accuracy', 0.0) # Get accuracy if available

    #         # Create a distinct name for saving
    #         save_name = f"gap_f1_best_{rank+1}_{name}"

    #         print(f"  Saving Rank {rank+1}: {save_name} (Gap: {top_gap:.4f}, F1: {top_f1:.4f})")

    #         # Save the model with the new name and relevant stats
    #         save_best_model(top_model, save_name, {
    #             'original_model_name': name,
    #             'rank_gap_f1': rank + 1,
    #             'cv_f1_score': top_f1,
    #             'cv_f1_gap': top_gap,
    #             'cv_accuracy': top_acc,
    #             'is_binary': is_binary,
    #             'preserve_zones': preserve_zones,
    #             'sort_features': sort_features,
    #             'normalization': normalization,
    #             'sampling_method': sampling_method, # Record sampling used during *tuning* or *global* application
    #             'num_features': X.shape[1],
    #             'selection_criteria': 'min_gap_then_max_f1_cv'
    #         })
    #     # ------------

    # Perform SHAP analysis on the best model if requested
    if analyze_shap and best_model is not None:
        print(f"\nPerforming SHAP analysis on the best model ({best_model_name})...")
        
        
        # Generate feature names if needed
        # if feature_names is None:
        #     metrics = ['mean', 'median', 'std', 'iqr', 'idr', 'skew', 'kurt', 'Del', 'Amp']
        #     feature_names = [f'{m}_Z{i+1}' for m in metrics for i in range(20)]
        
        metrics_shap = ['mean', 'median', 'std', 'iqr', 'idr', 'skew', 'kurt', 'Del', 'Amp'] 
        current_feature_names_shap = [f'{m}_Z{i+1}' for m in metrics_shap for i in range(20)]
        if feature_names is None: feature_names = current_feature_names_shap
        
        # Run different SHAP visualizations
        try:
            # Basic visualizations for all model types
            run_shap_analysis(
                best_model_instance_overall, 
                X_test, 
                feature_names=feature_names, 
                plot_type='summary', 
                report_dir=report_dir, 
                is_binary=is_binary,
                y_test=y_test
            )
            
            run_shap_analysis(
                best_model_instance_overall, 
                X_test, 
                feature_names=feature_names, 
                plot_type='bar', 
                report_dir=report_dir, 
                is_binary=is_binary,
                y_test=y_test
            )
            
            # Additional visualizations for tree-based models
            # if isinstance(best_model,
            if isinstance(best_model_instance_overall,
                          (RandomForestClassifier, GradientBoostingClassifier, xgb.XGBClassifier)):
                run_shap_analysis(
                    best_model_instance_overall, 
                    X_test, 
                    feature_names=feature_names, 
                    plot_type='beeswarm', 
                    report_dir=report_dir, 
                    is_binary=is_binary,
                    y_test=y_test
                )
            
            # Sample-specific visualizations for different samples
            # First sample
            sample_idx = 0
            run_shap_analysis(best_model_instance_overall, X_test, feature_names, sample_idx=sample_idx, plot_type='waterfall', report_dir=report_dir, 
                is_binary=is_binary,
                y_test=y_test)
            run_shap_analysis(best_model_instance_overall, X_test, feature_names, sample_idx=sample_idx, plot_type='force', report_dir=report_dir, 
                is_binary=is_binary,
                y_test=y_test)
            
            # Second sample if available
            if len(X_test) > 1:
                run_shap_analysis(
                    best_model_instance_overall, 
                    X_test, 
                    feature_names=feature_names, 
                    sample_idx=1, 
                    plot_type='waterfall', 
                    report_dir=report_dir, 
                    is_binary=is_binary,
                    y_test=y_test
                    )
            
            # Random sample from minority class if this is a binary classification
            if is_binary and len(X_test) > 5:
                # Find indices of minority class samples
                minority_idx = np.where(y_test == np.bincount(y_test).argmin())[0]
                if len(minority_idx) > 0:
                    # Select a random minority class sample
                    sample_idx = np.random.choice(minority_idx)
                    run_shap_analysis(
                        best_model_instance_overall, 
                        X_test, 
                        feature_names=feature_names, 
                        sample_idx=sample_idx, 
                        plot_type='waterfall', 
                        report_dir=report_dir, 
                        is_binary=is_binary,
                        y_test=y_test
                            )
            
            # Try to analyze feature group importance
            try:
                # For tree-based models, can use the original feature group importance
                if isinstance(best_model_instance_overall, (RandomForestClassifier, GradientBoostingClassifier, xgb.XGBClassifier)):
                    analyze_feature_group_importance(
                        best_model_instance_overall, 
                        X_test, 
                        num_features=len(feature_indices) if feature_indices is not None else 9,
                        metrics=metrics_shap if 'metrics_shap' in locals() else None,
                        class_index=0, report_dir=report_dir, y_test=y_test
                    )
                else:
                    # For non-tree models, we won't use feature group importance
                    print("Feature group importance analysis is skipped for non-tree models")
            except Exception as e:
                print(f"Warning when analyzing feature group importance: {e}")
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error during SHAP analysis: {e}")
            traceback.print_exc()
        
    #     # Add more analyses based on model type
    #     if best_model_name in ['Stacking Classifier', 'SVM_pipeline']:
    #         # For complex models, just do the basic analyses
    #         run_shap_analysis(best_model, X_test, feature_names, sample_idx=0, plot_type='waterfall')
    #     elif best_model_name in ['Random Forest', 'Gradient Boosting', 'xgb_model']:
    #         # For tree-based models, do more detailed analyses
    #         run_shap_analysis(best_model, X_test, feature_names, plot_type='beeswarm')
    #         run_shap_analysis(best_model, X_test, feature_names, sample_idx=0, plot_type='waterfall')
    #         run_shap_analysis(best_model, X_test, feature_names, sample_idx=0, plot_type='force')
    
    # # if analyze_shap and best_model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
    # #     print(f"\nPerforming SHAP analysis on {best_model_name}...")
    #         #     # Generate feature names
    #     # metrics = ['mean', 'median', 'std', 'iqr', 'idr', 'skew', 'kurt', 'Del', 'Amp']
    #     # feature_names = [f'{m}_Z{i+1}' for m in metrics for i in range(20)]
      
    # #     # Summary plot
        # run_shap_analysis(best_model, X_test, feature_names, plot_type='summary')
        #         # Bar plot
        # run_shap_analysis(best_model, X_test, feature_names, plot_type='bar')
        #         # Beeswarm plot
        # run_shap_analysis(best_model, X_test, feature_names, plot_type='beeswarm')
        #         # Waterfall plot for first sample
        # run_shap_analysis(best_model, X_test, feature_names, sample_idx=0, plot_type='waterfall')
        #         # Force plot for first sample
        # run_shap_analysis(best_model, X_test, feature_names, sample_idx=0, plot_type='force')
    
        # Analyze feature group importance
            # analyze_feature_group_importance(best_model, X_test, num_features=len(feature_indices) if feature_indices else 9)
        # Analyze feature group importance
        
    if best_model_instance_overall: # [cite: 897]
        num_feat_groups = len(feature_indices) if feature_indices else 9 # [cite: 897]
        metrics_group_analysis = ['mean', 'median', 'std', 'iqr', 'idr', 'skew', 'kurt', 'Del', 'Amp'] # Default if not generated earlier
        if is_binary:
            analyze_feature_group_importance(best_model_instance_overall, X_test, num_features=num_feat_groups, metrics=metrics_group_analysis, class_index=0, report_dir=report_dir, y_test=y_test) # [cite: 897]
        
        # if is_binary:
        #     analyze_feature_group_importance(best_model, X_test, num_features=len(feature_indices) if feature_indices else 9, class_index=0, report_dir=report_dir, y_test=y_test)
            
            
        else:   
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # Create a 2x2 grid of subplots

            # Plot for class index 0
            plt.sca(axs[0, 0])
            analyze_feature_group_importance(best_model_instance_overall, X_test, class_index=0,report_dir=report_dir, y_test=y_test)
            plt.title('Class 0')

            # Plot for class index 1
            plt.sca(axs[0, 1])
            analyze_feature_group_importance(best_model_instance_overall, X_test, class_index=1, report_dir=report_dir, y_test=y_test)
            plt.title('Class 1')

            # Plot for class index 2
            plt.sca(axs[1, 0])
            analyze_feature_group_importance(best_model_instance_overall, X_test, class_index=2, report_dir=report_dir, y_test=y_test)
            plt.title('Class 2')

            # Plot for class index 3
            plt.sca(axs[1, 1])
            analyze_feature_group_importance(best_model_instance_overall, X_test, class_index=3, report_dir=report_dir, y_test=y_test)
            plt.title('Class 3')

            plt.tight_layout()
            plt.show()
            plt.close()
            
    # # After your SHAP analysis is complete, add:
    #     # Get number of classes
    #     if hasattr(model, 'coef_') and len(model.coef_.shape) > 1:
    #         num_classes = model.coef_.shape[0]
    #     else:
    #         num_classes = 1  # Binary classification or tree-based model

    #     # Analyze top feature groups for each class
    #     for class_idx in range(num_classes):
    #         print(f"\n=== Analyzing Class {class_idx} ===")
            
    #         # Analyze median features
    #         # med_zones = analyze_feature_group_zones(
    #         #     best_model,
    #         #     X_test,
    #         #     feature_names,
    #         #     group_name="med",
    #         #     report_dir=report_dir,
    #         #     class_index=class_idx
    #         # )
            
    #         # Analyze skew features
    #         feat_zones = analyze_feature_group_zones(
    #             model,
    #             X_test,
    #             feature_names,
    #             group_name="skew",
    #             report_dir=report_dir,
    #             class_index=class_idx
    #         )
    
    # Add cross-validation evaluation for the best model
    # cv_results = evaluate_model_with_cv(
    #     best_model, X, y, best_model_name, n_splits=10, random_state=42
    # )
    
    final_cv_results = None
    if best_model_instance_overall: 
        final_cv_results = evaluate_model_with_cv( 
            best_model_instance_overall, X, y, best_model_name, n_splits=5, random_state=42)

    best_model_f1_score = f1_scores.get(best_model_name, 0.0) 
    best_model_roc_auc = roc_auc_scores.get(best_model_name, 0.0)
    best_model_avg_precision = avg_precision_scores.get(best_model_name, 0.0)

    # Ensure all expected keys are in the return dict for integrated_pipeline
    return_results_dict = {
        'trained_models': trained_models,
        'accuracies': accuracies,
        'f1_scores': f1_scores, # Dict of all F1s
        'roc_auc_scores': roc_auc_scores, # Dict of all ROC AUCs
        'avg_precision_scores': avg_precision_scores, # Dict of all APs
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'le': le,
        'feature_names': feature_names,
        'best_model_name': best_model_name if best_model_name else "N/A",
        'f1_score': best_model_f1_score, # Single F1 for best model
        'roc_auc': best_model_roc_auc, # Single ROC AUC for best model
        'avg_precision': best_model_avg_precision, # Single AP for best model
        'cv_results': final_cv_results
    }
    print(f"DEBUG run_classification_pipeline: Returning best_model_name: {return_results_dict['best_model_name']}") # [cite: 908]
    print(f"DEBUG run_classification_pipeline: Returning f1_score for best model: {return_results_dict['f1_score']}") # [cite: 908]

    
        
    # return best_model, {
    #     'trained_models': trained_models,
    #     # 'final_model_evaluations': final_model_evals,
    #     'accuracies': accuracies,
    #     'f1_scores': f1_scores,
    #     'X_train': X_train,
    #     'X_test': X_test,
    #     'y_train': y_train,
    #     'y_test': y_test,
    #     'le': le,
    #     'feature_names': feature_names,
    #     'best_model_name': best_model_name, # added
    #     'f1_score': best_model_f1_score,    # added
    #     'cv_results': cv_results if 'cv_results' in locals() else None
    # }
    
    # best_model_f1_score = f1_scores.get(best_model_name, 0.0) #  F1-> best model
    # print(f"DEBUG run_classification_pipeline: Returning best_model_name: {best_model_name}")
    # print(f"DEBUG run_classification_pipeline: Returning f1_score for best model: {best_model_f1_score}")
    # print(f"DEBUG run_classification_pipeline: Returning X_train shape: {X_train.shape}")
    # print(f"DEBUG run_classification_pipeline: Returning y_train shape: {y_train.shape}, unique: {np.unique(y_train)}")


    # return best_model, {
    #     'trained_models': trained_models,
    #     'accuracies': accuracies,
    #     'f1_scores': f1_scores, # This is the DICTIONARY of all F1 scores
    #     'X_train': X_train,  # Training features (post-sampling if applicable)
    #     'y_train': y_train,  # Training labels (post-sampling if applicable)
    #     'X_test': X_test,
    #     'y_test': y_test,
    #     'le': le,
    #     'feature_names': feature_names,
    #     'best_model_name': best_model_name, 
    #     'f1_score': best_model_f1_score,        # <<< THIS IS THE SINGLE F1 SCORE FOR THE BEST MODEL
    #     'cv_results': cv_results if 'cv_results' in locals() else None
    # }
    return best_model_instance_overall, return_results_dict


# Run a batch of experiments
def run_experiment_batch(data_path, experiment_configs=None):
    """
    Run a batch of experiments with different configurations
    
    Args:
        data_path: Path to data
        experiment_configs: List of experiment configurations
        
    Returns:
        best_overall_model: Best overall model
        
        Default configurations options:
            # 'normalization':   'standard', 'minmax', 'robust', 'none'
            # 'sampling_method': 'none', 'smote', 'random_over', 'random_under'
            # 'is_binary':       True for binary, False for multiclass
            # 'preserve_zones':  Keep spatial relationship between zones
            # 'sort_features':   'none', 'ascend_all', 'descend_all', 'custom'
            # 'tune_hyperparams':  Whether to tune hyperparameters
    """
    if experiment_configs is None:
        experiment_configs = [        
            # Binary classification experiments
            {'is_binary': True, 'normalization': 'none', 'sampling_method': 'smote'},
            {'is_binary': True, 'normalization': 'standard', 'sampling_method': 'smote'},
            {'is_binary': True, 'normalization': 'minmax', 'sampling_method': 'smote'},
            {'is_binary': True, 'normalization': 'robust', 'sampling_method': 'smote'},
            
            # Binary classification with feature sorting
            {'is_binary': True, 'normalization': 'standard', 'sampling_method': 'smote', 'sort_features': 'ascend_all'},
            {'is_binary': True, 'normalization': 'standard', 'sampling_method': 'smote', 'sort_features': 'descend_all'},
            {'is_binary': True, 'normalization': 'standard', 'sampling_method': 'smote', 'sort_features': 'custom'},
            
            # Multiclass classification experiments
            {'is_binary': False, 'normalization': 'standard', 'sampling_method': 'none'},
            {'is_binary': False, 'normalization': 'standard', 'sampling_method': 'smote'},
            {'is_binary': False, 'normalization': 'minmax', 'sampling_method': 'smote'},
            {'is_binary': False, 'normalization': 'robust', 'sampling_method': 'smote'},
            
            # Multiclass classification with feature sorting
            {'is_binary': False, 'normalization': 'standard', 'sampling_method': 'smote', 'sort_features': 'ascend_all'},
            {'is_binary': False, 'normalization': 'standard', 'sampling_method': 'smote', 'sort_features': 'descend_all'},
            {'is_binary': False, 'normalization': 'standard', 'sampling_method': 'smote', 'sort_features': 'custom'}
        ]
    
    best_models = {}
    best_f1_scores = {}
    
    for i, config in enumerate(experiment_configs):
        print(f"\n{'='*80}")
        print(f"Experiment {i+1}/{len(experiment_configs)}")
        print(f"Configuration: {config}")
        print(f"{'='*80}")
        
        # Extract configuration parameters
        is_binary = config.get('is_binary', True)
        normalization = config.get('normalization', 'standard')
        sampling_method = config.get('sampling_method', 'none')
        preserve_zones = config.get('preserve_zones', True)
        sort_features = config.get('sort_features', False)
        selected_features = config.get('selected_features', None)
        tune_hyperparams = config.get('tune_hyperparams', False)
        transform_features = config.get('transform_features', False)
        
        # Run classification pipeline
        try:
            best_model, results = run_classification_pipeline(
                data_path=data_path,
                normalization=normalization,
                sampling_method=sampling_method,
                is_binary=is_binary,
                preserve_zones=preserve_zones,
                sort_features=sort_features,
                selected_features=selected_features,
                tune_hyperparams=tune_hyperparams,
                analyze_shap=False,  # Skip SHAP analysis for batch experiments
                analyze_minkowski_dist=False  # Skip Minkowski analysis for batch experiments
            )
            
            # Get best model name and F1 score
            best_model_name = max(results['f1_scores'], key=results['f1_scores'].get)
            best_f1_score = results['f1_scores'][best_model_name]
            
            # Store results for this configuration
            config_key = f"{'binary' if is_binary else 'multiclass'}_{normalization}_{sampling_method}"
            if sort_features:
                config_key += "_sorted"
            
            best_models[config_key] = best_model
            best_f1_scores[config_key] = best_f1_score
            
            print(f"\nBest model for configuration {config_key}: {best_model_name}")
            print(f"F1 Score: {best_f1_score:.4f}")
            
        except Exception as e:
            print(f"Error in experiment {i+1}: {str(e)}")
    
    # Find best overall model
    if best_f1_scores:
        best_overall_config = max(best_f1_scores, key=best_f1_scores.get)
        best_overall_model = best_models[best_overall_config]
        best_overall_f1 = best_f1_scores[best_overall_config]
        
        print(f"\n{'='*80}")
        print(f"Best overall configuration: {best_overall_config}")
        print(f"Best overall F1 Score: {best_overall_f1:.4f}")
        print(f"{'='*80}")
        
        # Run SHAP analysis on the best overall model
        print("\nRunning SHAP analysis on the best overall model...")
        data = pd.read_csv(data_path)
        
        # Extract configuration from key
        config_parts = best_overall_config.split('_')
        is_binary = config_parts[0] == 'binary'
        normalization = config_parts[1]
        sampling_method = config_parts[2]
        sort_features = len(config_parts) > 3 and config_parts[3] == 'sorted'
        
        # Prepare data for SHAP analysis
        X, y, le, scaler, feature_names, actual_group_metrics = prepare_data(
                                    data, 
                                    normalization=normalization, 
                                    is_binary=is_binary,
                                    preserve_zones=preserve_zones,
                                    sort_features=sort_features,
                                    transform_features=transform_features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Generate feature names
        metrics = ['mean', 'median', 'std', 'iqr', 'idr', 'skew', 'kurt', 'Del', 'Amp']
        feature_names = [f'{m}_Z{i+1}' for m in metrics for i in range(20)]
        
        # Run SHAP analysis
        run_shap_analysis(best_overall_model, X_test, feature_names, plot_type='summary', 
                is_binary=is_binary,
                y_test=y_test)
        run_shap_analysis(best_overall_model, X_test, feature_names, plot_type='bar', 
                is_binary=is_binary,
                y_test=y_test)
        
        # Analyze feature group importance
        analyze_feature_group_importance(best_overall_model, X_test)
        
        return best_overall_model
    else:
        print("No successful experiments found")
        return None
    
# Function to generate all combinations of parameters
def generate_parameter_combinations():
    """
    Generate all combinations of parameters for grid search
    
    Returns:
        List of dictionaries with all parameter combinations
    """
    from itertools import product
    
    # Define parameter grid
    param_grid = {
        'is_binary': [True, False],
        'normalization': ['none', 'standard', 'minmax', 'robust'],
        'sampling_method': ['none', 'smote', 'random_over', 'random_under'],
        'preserve_zones': [True, False],
        # 'selected_features': None,  # List of selected features (None for all)
        'feature_indices': [3, 4, 8],  # Select only features 4, 5, and 9 (0-indexed)
        'sort_features': [
            'none',                   # No sorting applied
            'ascend_all',             # Sort all features ascending
            'descend_all',            # Sort all features descending
            # 'custom',                 # Use predefined custom sorting in prepare_data
            # You can also add dictionary-based custom sorting options:
            {'ascending': [0, 1, 3,4,6,8], 'descending': [2,5,7]}, #, 'unsorted': []},
            {'ascending': [2,5,7], 'descending': [0, 1, 3,4,6,8]},
        ],
        'tune_hyperparams': [False]  # Usually just keep as False for batch runs
    }
    
    # Get all keys and values
    keys = param_grid.keys()
    values = param_grid.values()
    
    # Generate all combinations
    combinations = list(product(*values))
    
    # Convert to list of dictionaries
    param_combinations = [dict(zip(keys, combo)) for combo in combinations]
    
    print(f"Generated {len(param_combinations)} parameter combinations")
    
    return param_combinations

# Example of usage in main script:
if __name__ == "__main__":
    # data_path = '/home/users/u5499379/Projects/ardes/mpod.csv'
    data_path = '/home/users/u5499379/Projects/ardes/ardes_ML_v1/data/train_mpod.csv'
    
    # Option 1: Run a single experiment with specific parameters
    params = {
        'data_path': data_path,
        'normalization': 'none',  # 'standard', 'minmax', 'robust', 'none'
        'sampling_method': 'smote', # 'none', 'smote', 'random_over', 'random_under'
        'is_binary': True,          # True / False for multiclass
        'preserve_zones': False,    # True / False
        'selected_features': None,  # List of selected features (None for all)
        'feature_indices': None, #[3, 4, 8],  # Select only features 4, 5, and 9 (0-indexed)
        'sort_features': 'descend_all',  # 'none', 'ascend_all', 'descend_all', 'custom'
        # 'sort_features': 'custom',# {'ascending': [0, 1, 3,4,6,8], 'descending': [2,5,7]},
        'tune_hyperparams': True
    }
    
    # Run single experiment
    best_model, results = run_classification_pipeline(**params)
    
    # Option 2: Run a subset of combinations
    # Define specific experiment configurations to run
    # experiment_configs = [
    #     {'is_binary': True, 'normalization': 'standard', 'sampling_method': 'smote', 'sort_features': 'ascend_all'},
    #     {'is_binary': True, 'normalization': 'standard', 'sampling_method': 'smote', 'sort_features': 'descend_all'},
    #     {'is_binary': True, 'normalization': 'minmax', 'sampling_method': 'smote', 'sort_features': 'custom'}
    # ]
    
    # # Run batch of specific experiments
    # best_overall_model = run_experiment_batch(data_path, experiment_configs)
    
    # Option 3: Run all combinations 
    # # Generate all parameter combinations
    # all_combinations = generate_parameter_combinations()
    
    # # If needed, limit the number of combinations to run
    # max_combinations = 10  # Adjust as needed
    # selected_combinations = all_combinations[:max_combinations]
    
    # # Run selected batch of experiments
    # best_overall_model = run_experiment_batch(data_path, selected_combinations)
    
    # Add this line at the very end:
    plt.close('all')
    print("Closed all matplotlib figures.")