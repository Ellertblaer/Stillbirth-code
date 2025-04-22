#Þessi kóði reyndist mér erfiður local, svo ég runnaði frekar dóti í Colab.

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder # Keep just in case, but likely not needed if all numeric
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Input files (the trimmed ones with common columns)
TRAIN_VAL_CSV_PATH = 'Stora_settid/combined_birth_data_2023_ml_ready_stora-FINAL.csv'
TEST_CSV_PATH = 'Litla_settid/combined_birth_data_2023_ml_ready_litla-FINAL.csv'
#Fyrir stóra settið
ratio_negatives_to_positives = 0.0001
#Fyrir litla settið
#ratio_negatives_to_positives = 0.0115

TARGET_COLUMN = 'Outcome'

# Split ratio for train/validation sets from the "little" dataset
VALIDATION_SIZE = 0.2 # Use 20% of the little dataset for validation
RANDOM_STATE = 42 # For reproducibility

# XGBoost Hyperparameters (Start with these, tune later)
XGB_PARAMS = {
    'objective': 'binary:logistic', # For binary classification
    'eval_metric': 'auc',          # Evaluation metric (Area Under ROC Curve) - good for imbalance
    'eta': 0.1,                    # Learning rate
    'max_depth': 6,                # Maximum depth of a tree
    'subsample': 0.8,              # Fraction of samples used per tree
    'colsample_bytree': 0.8,       # Fraction of features used per tree
    'gamma': 0,                    # Minimum loss reduction required to make a further partition
    'lambda': 1,                   # L2 regularization term on weights (xgb's default)
    'alpha': 0,                    # L1 regularization term on weights (xgb's default)
    'use_label_encoder': False,    # Recommended setting
    'random_state': RANDOM_STATE,
    # Add scale_pos_weight if dataset is highly imbalanced:
    'scale_pos_weight': ratio_negatives_to_positives,
    'n_estimators': 1000          # Start high, use early stopping
}
EARLY_STOPPING_ROUNDS = 50

print("--- Loading Data ---")

try:
    print(f"Loading Training/Validation data from: {TRAIN_VAL_CSV_PATH}")
    df_train_val = pd.read_csv(TRAIN_VAL_CSV_PATH, low_memory=False) # low_memory=False can help with mixed types
    print(f"Loaded {len(df_train_val)} training/validation records.")

    print(f"Loading Test data from: {TEST_CSV_PATH}")
    df_test = pd.read_csv(TEST_CSV_PATH, low_memory=False)
    print(f"Loaded {len(df_test)} test records.")

except FileNotFoundError as e:
    print(f"Error: Input CSV file not found. Please ensure these files exist:")
    print(f"- {TRAIN_VAL_CSV_PATH}")
    print(f"- {TEST_CSV_PATH}")
    print(f"Details: {e}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

print("\n--- Data Preprocessing ---")

# --- Handle Missing Values (Option 1: Let XGBoost handle NaNs) ---
# XGBoost can often handle np.nan directly for numeric features.
# We primarily need to ensure data types are correct.

# --- Convert Data Types ---
# Identify columns that *should* be numeric (based on your knowledge and previous steps)
# Exclude the target column for now
all_cols = df_train_val.columns.tolist()
feature_cols = [col for col in all_cols if col != TARGET_COLUMN]

print(f"Attempting to convert {len(feature_cols)} feature columns to numeric...")
potential_non_numeric = []

for df in [df_train_val, df_test]: # Apply to both dataframes
    for col in feature_cols:
        try:
            # Convert to numeric, turning errors into NaN (which XGBoost might handle)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            # This shouldn't happen often if read_csv worked, but good to catch
            print(f"  Could not convert column '{col}' during numeric conversion: {e}. Keeping as object.")
            potential_non_numeric.append(col)

# Convert target column to integer
for df in [df_train_val, df_test]:
    try:
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce').astype('Int64') # Use nullable Int
        if df[TARGET_COLUMN].isnull().any():
             print(f"Warning: Found missing values in Target Column '{TARGET_COLUMN}' after conversion. Dropping these rows.")
             df.dropna(subset=[TARGET_COLUMN], inplace=True)
    except Exception as e:
        print(f"Error converting target column '{TARGET_COLUMN}': {e}. Exiting.")
        exit()

if potential_non_numeric:
     print("\nWarning: The following columns could not be reliably converted to numeric.")
     print("They might be categorical strings. XGBoost might struggle with these directly.")
     print("Consider using LabelEncoding or OneHotEncoding if needed.")
     print(list(set(potential_non_numeric))) # Show unique non-numeric cols


# --- Define Features (X) and Target (y) ---
X = df_train_val[feature_cols]
y = df_train_val[TARGET_COLUMN]

X_test_final = df_test[feature_cols]
y_test_final = df_test[TARGET_COLUMN]

# --- Verify column alignment ---
if not X.columns.equals(X_test_final.columns):
    print("Error: Columns in training/validation set do not match columns in test set AFTER processing!")
    print("Training/Validation columns:", X.columns)
    print("Test columns:", X_test_final.columns)
    # Find differences
    train_only = set(X.columns) - set(X_test_final.columns)
    test_only = set(X_test_final.columns) - set(X.columns)
    if train_only: print("Columns only in Train/Val:", train_only)
    if test_only: print("Columns only in Test:", test_only)
    exit()

print("\n--- Splitting Training and Validation Data ---")
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=y # Stratify ensures proportion of outcomes is similar in train/val splits
)
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test_final.shape}")

# --- Handle Imbalance (Optional but Recommended) ---
# Calculate scale_pos_weight if needed
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
if pos_count > 0:
    scale_pos_weight = neg_count / pos_count
    print(f"\nCalculated scale_pos_weight for imbalance: {scale_pos_weight:.2f}")
    # Uncomment the line below to actually use it in XGB_PARAMS
    # XGB_PARAMS['scale_pos_weight'] = scale_pos_weight
else:
    print("Warning: No positive samples (Outcome=1) found in training data.")


print("\n--- Training XGBoost Model ---")

# Instantiate the XGBoost classifier
model = xgb.XGBClassifier(**XGB_PARAMS)

# Define the evaluation set for early stopping
eval_set = [(X_train, y_train), (X_val, y_val)]

# Train the model
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose=50 # Print evaluation results every 50 rounds
)

print("\n--- Evaluating Model ---")

# --- Validation Set Evaluation ---
print("\n--- Validation Set Performance ---")
y_pred_val = model.predict(X_val)
y_prob_val = model.predict_proba(X_val)[:, 1] # Probabilities for the positive class (1)

print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
print(f"Validation ROC AUC: {roc_auc_score(y_val, y_prob_val):.4f}")
print("Validation Classification Report:")
print(classification_report(y_val, y_pred_val))
print("Validation Confusion Matrix:")
print(confusion_matrix(y_val, y_pred_val))

# --- Test Set Evaluation (Using the "big" dataset) ---
print("\n--- Test Set Performance (Big Dataset) ---")
y_pred_test = model.predict(X_test_final)
y_prob_test = model.predict_proba(X_test_final)[:, 1] # Probabilities for the positive class (1)

print(f"Test Accuracy: {accuracy_score(y_test_final, y_pred_test):.4f}")
print(f"Test ROC AUC: {roc_auc_score(y_test_final, y_prob_test):.4f}")
print("Test Classification Report:")
print(classification_report(y_test_final, y_pred_test))
print("Test Confusion Matrix:")
print(confusion_matrix(y_test_final, y_pred_test))

# --- Plot ROC Curve (Example) ---
fpr_val, tpr_val, _ = roc_curve(y_val, y_prob_val)
fpr_test, tpr_test, _ = roc_curve(y_test_final, y_prob_test)

plt.figure(figsize=(8, 6))
plt.plot(fpr_val, tpr_val, label=f'Validation AUC = {roc_auc_score(y_val, y_prob_val):.2f}')
plt.plot(fpr_test, tpr_test, label=f'Test AUC = {roc_auc_score(y_test_final, y_prob_test):.2f}', linestyle='--')
plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve.png') # Save the plot
print("\nROC curve saved to roc_curve.png")
# plt.show() # Uncomment if you want to display the plot interactively

# --- Feature Importance (Optional) ---
try:
    print("\n--- Feature Importance ---")
    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    print("Top 20 Important Features:")
    print(feature_importances.nlargest(20))

    # Plot top N features
    N = 20
    plt.figure(figsize=(10, N/2)) # Adjust figure size
    feature_importances.nlargest(N).plot(kind='barh')
    plt.title(f'Top {N} Feature Importances')
    plt.gca().invert_yaxis() # Display most important at the top
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print(f"\nFeature importance plot saved to feature_importance.png")
    # plt.show() # Uncomment to display plot
except Exception as e:
    print(f"Could not generate feature importance plot: {e}")


print("\nScript finished.")