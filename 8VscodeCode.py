import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, roc_auc_score, roc_curve)
# from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# INPUTS: Use the FINAL trimmed files with common columns
TRAIN_VAL_SOURCE_CSV = '/content/summa.csv'
TEST_SOURCE_CSV = '/content/combined_birth_data_2023_ml_ready_litla-FINAL.csv'

TARGET_COLUMN = 'Outcome'
VALIDATION_SIZE = 0.2
RANDOM_STATE = 42

# XGBoost Hyperparameters (Keep as before, or adjust)
# Ensure ratio_negatives_to_positives is appropriate for the *modified* training data ratio
# You might want to recalculate it after removing half the stillbirths from training
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.1, 'max_depth': 6, 'subsample': 0.8,'colsample_bytree': 0.8,
    'gamma': 0, 'lambda': 1, 'alpha': 0,
    'use_label_encoder': False,
    'random_state': RANDOM_STATE,
    # 'scale_pos_weight': calculated_value, # Calculate AFTER modifying y_train
    'n_estimators': 1000,
    'early_stopping_rounds': 50
}

print("--- Loading Data ---")
# Load the full source datasets first
try:
    print(f"Loading Training/Validation source data from: {TRAIN_VAL_SOURCE_CSV}")
    df_train_val_source = pd.read_csv(TRAIN_VAL_SOURCE_CSV, low_memory=False)
    print(f"Loaded {len(df_train_val_source)} train/val source records.")

    print(f"Loading Test source data from: {TEST_SOURCE_CSV}")
    df_test_source = pd.read_csv(TEST_SOURCE_CSV, low_memory=False)
    print(f"Loaded {len(df_test_source)} test source records.")

except FileNotFoundError as e:
    print(f"Error: Input CSV file not found. Check paths:")
    print(f"- {TRAIN_VAL_SOURCE_CSV}")
    print(f"- {TEST_SOURCE_CSV}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print("\n--- Initial Data Preprocessing (Types & Target) ---")
all_cols = df_train_val_source.columns.tolist()
feature_cols = [col for col in all_cols if col != TARGET_COLUMN]
potential_non_numeric = []

# Preprocess both source dataframes first
for df in [df_train_val_source, df_test_source]:
    # Convert features to numeric (coerce errors to NaN)
    for col in feature_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception:
            potential_non_numeric.append(col) # Track columns that failed conversion

    # Convert target to nullable integer, drop rows if target becomes NaN
    try:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce').astype('Int64')
        initial_len = len(df)
        df.dropna(subset=[TARGET_COLUMN], inplace=True)
        if len(df) < initial_len:
            print(f"Dropped {initial_len - len(df)} rows with missing target in {id(df)}") # Use id to differentiate dfs
    except Exception as e:
        print(f"Error converting target column '{TARGET_COLUMN}': {e}. Exiting.")
        exit()

if potential_non_numeric:
     print("\nWarning: Check non-numeric columns:", list(set(potential_non_numeric)))


# --- Define Full Features/Target Before Splitting/Modifying ---
X_source = df_train_val_source[feature_cols]
y_source = df_train_val_source[TARGET_COLUMN]

X_test_full = df_test_source[feature_cols]
y_test_full = df_test_source[TARGET_COLUMN]

# --- Verify column alignment ---
if not X_source.columns.equals(X_test_full.columns):
    print("Error: Columns mismatch between train/val source and test source!")
    exit()


print("\n--- Splitting Training and Validation Data (from 'Litla' set) ---")
X_train, X_val, y_train, y_val = train_test_split(
    X_source, y_source,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_source # Stratify based on original distribution
)
print(f"Initial training set shape: {X_train.shape}")
print(f"Initial validation set shape: {X_val.shape}")
print(f"Initial test set shape: {X_test_full.shape}")


# --- Modify Training Data: Remove FIRST half of Stillbirths ---
print("\n--- Modifying Training Data ---")
# Find indices of stillbirths (Outcome=0) IN THE TRAINING SET
stillbirth_train_indices = y_train[y_train == 0].index
n_stillbirths_train = len(stillbirth_train_indices)
print(f"Found {n_stillbirths_train} stillbirths in the initial training set.")

if n_stillbirths_train > 1:
    # Calculate the midpoint index *within the list of stillbirth indices*
    midpoint_train = n_stillbirths_train // 2
    # Get the indices of the first half of stillbirths to REMOVE
    indices_to_drop_train = stillbirth_train_indices[:midpoint_train]
    print(f"Removing the first {len(indices_to_drop_train)} stillbirth records from the training set.")

    # Drop these rows from BOTH X_train and y_train
    X_train = X_train.drop(index=indices_to_drop_train)
    y_train = y_train.drop(index=indices_to_drop_train)
    print(f"Modified training set shape: {X_train.shape}")
elif n_stillbirths_train == 1:
     print("Only one stillbirth found in training data, cannot remove half. Keeping it.")
else:
     print("No stillbirths found in the initial training set to remove.")


# --- Modify Test Data: Remove SECOND half of Stillbirths ---
print("\n--- Modifying Test Data ---")
# Find indices of stillbirths (Outcome=0) IN THE TEST SET
stillbirth_test_indices = y_test_full[y_test_full == 0].index
n_stillbirths_test = len(stillbirth_test_indices)
print(f"Found {n_stillbirths_test} stillbirths in the initial test set.")

if n_stillbirths_test > 1:
    # Calculate the midpoint index *within the list of stillbirth indices*
    midpoint_test = n_stillbirths_test // 2
    # Get the indices of the second half of stillbirths to REMOVE
    indices_to_drop_test = stillbirth_test_indices[midpoint_test:]
    print(f"Removing the second {len(indices_to_drop_test)} stillbirth records from the test set.")

    # Drop these rows from BOTH X_test_full and y_test_full
    X_test_final = X_test_full.drop(index=indices_to_drop_test) # Assign to final test var
    y_test_final = y_test_full.drop(index=indices_to_drop_test) # Assign to final test var
    print(f"Modified test set shape: {X_test_final.shape}")
elif n_stillbirths_test == 1:
     print("Only one stillbirth found in test data, cannot remove half. Keeping it.")
     X_test_final = X_test_full # Use original if no change
     y_test_final = y_test_full
else:
     print("No stillbirths found in the initial test set to remove.")
     X_test_final = X_test_full # Use original if no change
     y_test_final = y_test_full


# --- Recalculate scale_pos_weight based on MODIFIED training data ---
print("\n--- Handling Class Imbalance (Post-Modification) ---")
neg_count_train_mod = (y_train == 0).sum() # Use modified y_train
pos_count_train_mod = (y_train == 1).sum() # Use modified y_train

print(f"MODIFIED Training Counts: Outcome=0: {neg_count_train_mod}, Outcome=1: {pos_count_train_mod}")

if pos_count_train_mod > 0 and neg_count_train_mod > 0:
    scale_pos_weight = neg_count_train_mod / pos_count_train_mod
    print(f"Calculated scale_pos_weight for MODIFIED training data: {scale_pos_weight:.4f}")
    XGB_PARAMS['scale_pos_weight'] = scale_pos_weight
    print("Added scale_pos_weight to XGBoost parameters.")
elif 'scale_pos_weight' in XGB_PARAMS:
    # Remove the param if it can't be calculated to avoid errors/using old values
    del XGB_PARAMS['scale_pos_weight']
    print("Removed scale_pos_weight as one class count is zero in modified training data.")


print("\n--- Training XGBoost Model ---")
# --- Initialize and Train Model (Using modified X_train, y_train) ---
model = xgb.XGBClassifier(**XGB_PARAMS)
eval_set = [(X_val, y_val)] # Evaluate on the unmodified validation set

model.fit(
    X_train, y_train, # Fit on modified training data
    eval_set=eval_set,
    verbose=50
    # early_stopping_rounds is now in XGB_PARAMS
)

print("\n--- Evaluating Model ---")

# --- Validation Set Evaluation (Unmodified Validation Set) ---
print("\n--- Validation Set Performance ---")
y_pred_val = model.predict(X_val)
y_prob_val = model.predict_proba(X_val)[:, 1]
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
print(f"Validation ROC AUC: {roc_auc_score(y_val, y_prob_val):.4f}")
print("Validation Classification Report:")
print(classification_report(y_val, y_pred_val, zero_division=0)) # Add zero_division
print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_val))

# --- Test Set Evaluation (Modified Test Set) ---
print("\n--- Test Set Performance (Modified Test Set) ---")
y_pred_test = model.predict(X_test_final) # Use modified X_test_final
y_prob_test = model.predict_proba(X_test_final)[:, 1]
print(f"Test Accuracy: {accuracy_score(y_test_final, y_pred_test):.4f}") # Use modified y_test_final
print(f"Test ROC AUC: {roc_auc_score(y_test_final, y_prob_test):.4f}") # Use modified y_test_final
print("Test Classification Report:")
print(classification_report(y_test_final, y_pred_test, zero_division=0)) # Use modified y_test_final
print("Test Confusion Matrix:")
print(confusion_matrix(y_test_final, y_pred_test)) # Use modified y_test_final

# --- Plot ROC Curve (Using modified y_test_final) ---
# ... (Plotting code remains largely the same, just ensure using y_test_final) ...
fpr_val, tpr_val, _ = roc_curve(y_val, y_prob_val)
fpr_test, tpr_test, _ = roc_curve(y_test_final, y_prob_test) # Use modified y_test_final

plt.figure(figsize=(8, 6))
plt.plot(fpr_val, tpr_val, label=f'Validation AUC = {roc_auc_score(y_val, y_prob_val):.2f}')
plt.plot(fpr_test, tpr_test, label=f'Test AUC = {roc_auc_score(y_test_final, y_prob_test):.2f}', linestyle='--') # Use modified y_test_final
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve_modified_splits.png') # Changed filename
print("\nROC curve saved to roc_curve_modified_splits.png")

# --- Feature Importance ---
# ...(feature importance code remains the same)...
try:
    print("\n--- Feature Importance ---")
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns) # Importance based on modified train
    print("Top 20 Important Features:")
    print(feature_importances.nlargest(20))
    N = 20
    plt.figure(figsize=(10, N/2))
    feature_importances.nlargest(N).plot(kind='barh')
    plt.title(f'Top {N} Feature Importances (Trained on Modified Data)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_modified_splits.png') # Changed filename
    print(f"\nFeature importance plot saved to feature_importance_modified_splits.png")
except Exception as e:
    print(f"Could not generate feature importance plot: {e}")


print("\nScript finished.")