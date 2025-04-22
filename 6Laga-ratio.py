import pandas as pd
import numpy as np
import os
import math
import random # For random sampling decision

# --- Configuration ---
# Input file containing BOTH live births (1) and stillbirths (0)
#Því remove same gerði ekki neitt snjallt.
#INPUT_CSV_PATH = 'Take 2/4seinni-filter/summa.csv' # Your combined file
INPUT_CSV_PATH = 'Litla_settid/combined_birth_data_2023_ml_ready_litla-FINAL.csv' # Your combined file

# Output file for the downsampled data
#OUTPUT_DOWNSAMPLED_CSV = 'Take 2/6laga-ratio/training.csv'
OUTPUT_DOWNSAMPLED_CSV = 'Take 2/6laga-ratio/testing.csv'

# Target column name
TARGET_COLUMN = 'Outcome'

# Desired ratio (Live Births : Stillbirths)
DESIRED_RATIO = 10

# Random state for reproducibility of sampling
RANDOM_STATE = 42
random.seed(RANDOM_STATE) # Seed the random module
np.random.seed(RANDOM_STATE) # Seed numpy's random module too

# Chunk size for reading/writing large files
CHUNK_SIZE = 500000 # Adjust based on memory

# Directory configuration
INPUT_OUTPUT_DIR = '.' # Assume files in current dir, adjust if needed

# Construct full paths
input_path = os.path.join(INPUT_OUTPUT_DIR, INPUT_CSV_PATH)
output_path = os.path.join(INPUT_OUTPUT_DIR, OUTPUT_DOWNSAMPLED_CSV)

# --- Pass 1: Count Outcomes ---
print(f"--- Pass 1: Counting outcomes in {input_path} ---")
total_rows = 0
n_stillbirths = 0
n_livebirths = 0
actual_header = None

try:
    reader = pd.read_csv(input_path, chunksize=CHUNK_SIZE, low_memory=False, dtype=str) # Read as string for counting
    for i, chunk_df in enumerate(reader):
        if i == 0:
            actual_header = chunk_df.columns.tolist() # Capture header from first chunk

        # Make sure target column exists
        if TARGET_COLUMN not in chunk_df.columns:
             raise ValueError(f"Target column '{TARGET_COLUMN}' not found in chunk {i+1}.")

        # Convert target to numeric for counting (handle errors)
        outcome_col = pd.to_numeric(chunk_df[TARGET_COLUMN], errors='coerce')

        n_stillbirths += (outcome_col == 0).sum()
        n_livebirths += (outcome_col == 1).sum()
        total_rows += len(chunk_df)
        print(f"  Processed chunk {i+1}, current Stillbirths: {n_stillbirths:,}, current Live Births: {n_livebirths:,}")

    print("\n--- Pass 1 Complete ---")
    print(f"Total rows read: {total_rows:,}")
    print(f"Total Stillbirths (Outcome=0) found: {n_stillbirths:,}")
    print(f"Total Live Births (Outcome=1) found: {n_livebirths:,}")

except FileNotFoundError:
    print(f"Error: Input file not found at '{input_path}'.")
    exit()
except ValueError as e:
     print(f"Error during Pass 1: {e}")
     exit()
except Exception as e:
    print(f"An error occurred during Pass 1: {e}")
    exit()

# --- Calculate Sampling Parameters ---
if n_stillbirths == 0:
    print("\nWarning: No stillbirths found. Cannot apply ratio. Consider copying the original file or stopping.")
    # Decide action: Maybe copy the whole file? Or exit? For now, we'll proceed and only write live births (no downsampling).
    keep_prob = 1.0 # Keep all live births
    target_live_count = n_livebirths
    print("Proceeding to write all found live births.")
elif n_livebirths == 0:
     print("\nWarning: No live births found. Output will only contain stillbirths.")
     keep_prob = 0.0 # Keep no live births (though none exist)
     target_live_count = 0
else:
    target_live_count = math.ceil(n_stillbirths * DESIRED_RATIO)
    keep_prob = min(1.0, target_live_count / n_livebirths) # Cap probability at 1.0
    print(f"\nTarget number of live births to keep (approx): {target_live_count:,}")
    print(f"Probability of keeping a live birth record: {keep_prob:.6f}")


# --- Pass 2: Filter and Write ---
print(f"\n--- Pass 2: Reading '{input_path}', applying filter, and writing to '{output_path}' ---")
lines_written = 0
live_births_kept_count = 0
stillbirths_kept_count = 0
first_chunk_written = False

try:
    # Read again, process, write
    reader = pd.read_csv(input_path, chunksize=CHUNK_SIZE, low_memory=False, dtype=str) # Keep as string initially

    # Open output file in write mode ('w')
    with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
        for i, chunk_df in enumerate(reader):
            print(f"  Processing chunk {i+1}...")

            if TARGET_COLUMN not in chunk_df.columns:
                print(f"Error: Target column '{TARGET_COLUMN}' missing in chunk {i+1}. Skipping chunk.")
                continue

            # Prepare outcome column for filtering
            outcome_col = pd.to_numeric(chunk_df[TARGET_COLUMN], errors='coerce')

            # Identify rows based on outcome
            is_stillbirth = (outcome_col == 0)
            is_livebirth = (outcome_col == 1)
            is_valid_outcome = is_stillbirth | is_livebirth # Rows where outcome is 0 or 1

            # Select valid stillbirths
            chunk_stillbirths = chunk_df[is_stillbirth & is_valid_outcome] # Ensure outcome is valid 0
            stillbirths_kept_count += len(chunk_stillbirths)

            # Select valid live births and apply sampling probability
            chunk_livebirths = chunk_df[is_livebirth & is_valid_outcome] # Ensure outcome is valid 1
            if not chunk_livebirths.empty and keep_prob < 1.0:
                 # Create a random mask based on keep_prob
                 keep_mask = np.random.rand(len(chunk_livebirths)) < keep_prob
                 chunk_livebirths_kept = chunk_livebirths[keep_mask]
            elif not chunk_livebirths.empty and keep_prob >= 1.0:
                 chunk_livebirths_kept = chunk_livebirths # Keep all if probability is 1
            else:
                 chunk_livebirths_kept = pd.DataFrame(columns=chunk_df.columns) # Empty DF if no live births

            live_births_kept_count += len(chunk_livebirths_kept)

            # Combine kept records for this chunk
            chunk_to_write = pd.concat([chunk_stillbirths, chunk_livebirths_kept])

            if not chunk_to_write.empty:
                 # Write chunk to CSV
                 chunk_to_write.to_csv(
                     f_out,
                     header=not first_chunk_written, # Write header only if it's the very first write
                     index=False,
                     lineterminator='\n'
                 )
                 first_chunk_written = True # Mark header as written
                 lines_written += len(chunk_to_write)

    print("\n--- Pass 2 Complete ---")
    print(f"Total unique lines written to '{output_path}': {lines_written:,}")
    print(f"  Stillbirths kept: {stillbirths_kept_count:,}")
    print(f"  Live births kept: {live_births_kept_count:,}")
    if stillbirths_kept_count > 0:
        final_ratio = live_births_kept_count / stillbirths_kept_count
        print(f"  Achieved Ratio (Live/Stillbirth): {final_ratio:.2f}:1")
    elif live_births_kept_count > 0:
         print("  Ratio not applicable (no stillbirths).")


except FileNotFoundError:
    print(f"Error: Input file not found during Pass 2: '{input_path}'.")
except Exception as e:
    print(f"An error occurred during Pass 2: {e}")

print("\nScript finished.")