import pandas as pd

# --- Configuration ---
# NOTE: You are using the "stora" (big) dataset path here.
# Usually, scale_pos_weight is calculated on the TRAINING data (part of the "litla" set).
# Make sure this path is intentionally the one you want to calculate the ratio from.
# If you meant to use the smaller set for training ratio, change this path back.
TRAIN_VAL_CSV_PATH = 'Litla_settid\combined_birth_data_2023_ml_ready_litla-FINAL.csv'
TARGET_COLUMN = 'Outcome'

print(f"--- Calculating Ratio for {TARGET_COLUMN} ---")

try:
    print(f"Loading target column from: {TRAIN_VAL_CSV_PATH}")
    # Load the target column - squeeze is removed
    df_target = pd.read_csv(TRAIN_VAL_CSV_PATH, usecols=[TARGET_COLUMN], low_memory=False) # Keep low_memory=False for potential dtype issues

    # Explicitly select the column to get a Series
    if TARGET_COLUMN in df_target.columns:
        y_train_val = df_target[TARGET_COLUMN]
        print(f"Loaded {len(y_train_val)} records.")
    else:
        # This should not happen if usecols worked, but good safety check
        print(f"Error: Target column '{TARGET_COLUMN}' not found after loading.")
        exit()


    # Ensure the target is treated as numeric/integer for counting
    # Convert potential strings '0'/'1' to integers, handle errors
    y_train_val = pd.to_numeric(y_train_val, errors='coerce')
    y_train_val.dropna(inplace=True) # Remove rows where outcome couldn't be parsed
    y_train_val = y_train_val.astype(int)

    # Calculate counts for Outcome = 0 (Stillbirths) and Outcome = 1 (Live Births)
    neg_count = (y_train_val == 0).sum()
    pos_count = (y_train_val == 1).sum()

    print(f"\nCounts in '{TRAIN_VAL_CSV_PATH}':")
    print(f"  Outcome=0 (Stillbirths): {neg_count}")
    print(f"  Outcome=1 (Live Births): {pos_count}")

    # Calculate the ratio: number of negatives / number of positives
    if pos_count > 0:
        scale_pos_weight = neg_count / pos_count
        print(f"\nCalculated scale_pos_weight (negatives / positives): {scale_pos_weight:.4f}")
    else:
        print("\nCannot calculate scale_pos_weight: No positive samples (Outcome=1) found.")

except FileNotFoundError:
    print(f"Error: File not found at {TRAIN_VAL_CSV_PATH}")
except KeyError:
    # This might catch issues if the column wasn't loaded correctly before selection
    print(f"Error: Target column '{TARGET_COLUMN}' potentially not found in {TRAIN_VAL_CSV_PATH}.")
except Exception as e:
    print(f"An error occurred: {e}")