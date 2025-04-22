import pandas as pd
import numpy as np
import os

# --- Configuration ---
# Input files (the ones created by the previous script)
NATALITY_INPUT_CSV = 'Take 2/3merged/merge.csv'
FETAL_DEATH_INPUT_CSV = 'Litla_settid/combined_birth_data_2023_ml_ready_litla-FINAL.csv'

# Output files (new files with only common columns)
NATALITY_TRIMMED_CSV = 'Take 2/4seinni-filter/summa.csv'
FETAL_DEATH_TRIMMED_CSV = 'Take 2/4seinni-filter/rusl.csv'

# Directory configuration (assuming files are in the same dir as the script)
INPUT_OUTPUT_DIR = '.'

# Chunk size for reading/writing large files (adjust if memory issues persist)
# If your individual processed CSVs definitely fit in memory, you could skip chunking here,
# but chunking is safer if you are unsure or might run this on different machines.
CHUNK_SIZE = 500000

# Construct full paths
natality_input_path = os.path.join(INPUT_OUTPUT_DIR, NATALITY_INPUT_CSV)
fetal_death_input_path = os.path.join(INPUT_OUTPUT_DIR, FETAL_DEATH_INPUT_CSV)
natality_output_path = os.path.join(INPUT_OUTPUT_DIR, NATALITY_TRIMMED_CSV)
fetal_death_output_path = os.path.join(INPUT_OUTPUT_DIR, FETAL_DEATH_TRIMMED_CSV)

# --- Step 1: Find Common Columns ---
print("--- Finding Common Columns ---")
try:
    print(f"Reading header from: {natality_input_path}")
    natality_cols = pd.read_csv(natality_input_path, nrows=0).columns.tolist()
    print(f"Reading header from: {fetal_death_input_path}")
    fetal_cols = pd.read_csv(fetal_death_input_path, nrows=0).columns.tolist()

    # Find the intersection using sets
    common_columns = sorted(list(set(natality_cols) & set(fetal_cols)))

    if not common_columns:
        print("Error: No common columns found between the two files. Exiting.")
        exit()

    print(f"\nFound {len(common_columns)} common columns:")
    # Print fewer columns if the list is very long
    if len(common_columns) > 30:
         print(common_columns[:15], "...", common_columns[-15:])
    else:
         print(common_columns)

except FileNotFoundError as e:
    print(f"Error: Input CSV file not found. Make sure these files exist:")
    print(f"- {natality_input_path}")
    print(f"- {fetal_death_input_path}")
    print(f"Details: {e}")
    exit()
except Exception as e:
    print(f"Error reading CSV headers: {e}")
    exit()

# --- Step 2: Function to Trim CSV to Common Columns (Chunked) ---
def trim_csv_to_common_columns(input_path, output_path, common_cols, chunk_size):
    """
    Reads input_path in chunks, keeps only common_cols, writes to output_path.
    """
    print(f"\n--- Trimming {input_path} -> {output_path} ---")
    try:
        first_chunk = True
        chunk_count = 0
        total_rows_processed = 0

        reader = pd.read_csv(
            input_path,
            chunksize=chunk_size,
            low_memory=False,
            on_bad_lines='skip' # Change to 'skip'
        )

        # Check if all common columns actually exist in this file's header
        # (Should be guaranteed by how common_columns was derived, but good check)
        input_header_cols = pd.read_csv(input_path, nrows=0).columns.tolist()
        actual_common_in_file = [col for col in common_cols if col in input_header_cols]
        if len(actual_common_in_file) != len(common_cols):
             missing = set(common_cols) - set(actual_common_in_file)
             print(f"  Warning: Not all 'common' columns were found in the header of {input_path}. Missing: {missing}")
             # Proceeding with columns that ARE actually present


        with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
            for chunk_df in reader:
                chunk_count += 1
                total_rows_processed += len(chunk_df)
                if chunk_count % 5 == 1 or chunk_size < 50000: # Reduce print frequency
                    print(f"  Processing chunk {chunk_count}...")

                # Select only the common columns that exist in this file
                trimmed_chunk = chunk_df[actual_common_in_file]

                # Write the trimmed chunk
                trimmed_chunk.to_csv(
                    f_out,
                    header=first_chunk, # Write header only for the first chunk
                    index=False,
                    lineterminator='\n'
                )
                first_chunk = False # Turn off header writing

        print(f"Finished trimming. Processed {total_rows_processed} rows.")
        print(f"Trimmed file saved to: {output_path}")
        return True

    except FileNotFoundError:
        print(f"Error: Input file not found during trimming: {input_path}")
        return False
    except Exception as e:
        print(f"An error occurred while trimming {input_path}: {e}")
        return False

# --- Step 3: Execute Trimming for Both Files ---
success1 = trim_csv_to_common_columns(
    input_path=natality_input_path,
    output_path=natality_output_path,
    common_cols=common_columns,
    chunk_size=CHUNK_SIZE
)

success2 = trim_csv_to_common_columns(
    input_path=fetal_death_input_path,
    output_path=fetal_death_output_path,
    common_cols=common_columns,
    chunk_size=CHUNK_SIZE
)

# --- Final Summary ---
if success1 and success2:
    print(f"\nSuccessfully created trimmed files:")
    print(f"- {natality_output_path}")
    print(f"- {fetal_death_output_path}")
    print("\nThese files now contain only the columns shared between the original processed CSVs.")
else:
    print("\nScript finished with errors during trimming.")

print("\nScript finished.")