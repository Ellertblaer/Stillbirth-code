import pandas as pd
import numpy as np
import os

# --- Configuration ---
# INPUTS: The CSVs created by the very FIRST script (parsing the raw TXT)
# Make sure these filenames are correct!
NATALITY_PROCESSED_CSV = 'Stora_settid/natality_2023_processed.csv' # Large Natality Processed CSV
FETAL_DEATH_PROCESSED_CSV = 'Stora_settid/fetal_deaths_2023_processed.csv' # Fetal Death Processed CSV

# OUTPUT: The SINGLE combined file you want for analysis/training
COMBINED_OUTPUT_CSV = 'Stora_settid/combined_birth_data_2023_ml_ready_stora2.csv' # Single output file name

# Directory configuration
INPUT_OUTPUT_DIR = '.' # Assume inputs/output in current dir, adjust if needed

# Directory configuration
# ***** ADD THIS LINE BACK *****
OUTPUT_DIR = '.' # Assume inputs/output in current dir, adjust if needed
# ****************************

# Construct full paths
natality_input_path = os.path.join(INPUT_OUTPUT_DIR, NATALITY_PROCESSED_CSV)
fetal_death_input_path = os.path.join(INPUT_OUTPUT_DIR, FETAL_DEATH_PROCESSED_CSV)
combined_output_path = os.path.join(INPUT_OUTPUT_DIR, COMBINED_OUTPUT_CSV)


# --- Define Chunk Size ---
CHUNK_SIZE = 500000

# --- Column Selection ---
# Define the columns you WANT in your FINAL combined dataset.
# These names MUST exist in the respective _processed.csv files.
# Standardize names if needed (like for RF_EHYPE vs RF_ECLAM -> RF_ECLAMPSIA)
# Make sure you fixed the first script if columns were missing (like Risk Factors)
relevant_columns_to_keep = [
    # Timing
    'DOB_WK', # Standardized name (Check if correct in both _processed files)

    # Location/Facility
    'BFACIL', 'BFACIL3',

    # Mother's Demographics
    'MAGER', 'MBSTATE_REC', 'RESTATUS', 'MRACE31',
    'MHISPX', 'MHISP_R', 'MRACEHISP', 'MEDUC',

    # Father's Demographics
    'FAGECOMB',

    # Pregnancy History
    'PRIORLIVE', 'PRIORDEAD', 'LBO_REC', 'ILLB_R',

    # Prenatal Care
    'PRECARE', 'PRECARE5', 'WIC', # Use PRECARE5

    # Maternal Health / Habits
    'CIG_0', 'CIG_1', 'CIG_2', 'CIG_3', 'CIG_REC',
    'M_Ht_In', 'BMI', 'BMI_R', 'PWgt_R',

    # Risk Factors (Use standardized names if you changed them in script 1)
    'RF_DIAB', 'RF_GEST', 'RF_PHYP', 'RF_GHYP',
    'RF_ECLAMPSIA', # Use the standardized name you chose
    'RF_INFTR', 'RF_FEDRG', 'RF_ARTEC', 'RF_CESAR', 'RF_CESARN',

    # Labor/Delivery Characteristics
    'ME_PRES', 'ME_ROUT', 'ME_TRIAL', 'RDMETH_REC', 'ATTEND', 'DPLURAL',

    # Infant Characteristics
    'SEX', 'COMBGEST', 'OEGest_Comb', 'DBWT',

    # Target Variable (Will be added)
    'Outcome'
]

# Define missing value codes
missing_value_codes = {
    '9', '99', '999', '9999', 'U', 'X', ''
}

# --- Function to Process and Append CSV Chunks ---
def process_and_append_chunked(input_csv, output_csv, outcome_val, cols_to_keep, missing_codes, chunk_size, write_header):
    """
    Reads an input CSV in chunks, processes, and appends to an output CSV.
    Selects only the columns specified in cols_to_keep that exist in the input file.
    """
    print(f"Processing {input_csv} -> {output_csv} (Outcome={outcome_val}, Header={write_header})")
    try:
        chunk_count = 0
        total_rows_processed = 0
        first_chunk_for_this_file = True

        # Check header of the input file once before iterating
        try:
            input_cols = pd.read_csv(input_csv, nrows=0, low_memory=False).columns.tolist()
            # Find which requested columns are ACTUALLY in this specific input file
            # Ensure 'Outcome' isn't required to be present yet
            actual_cols_to_use_in_file = [col for col in cols_to_keep if col in input_cols]
            missing_cols_in_file = set(cols_to_keep) - set(actual_cols_to_use_in_file) - {'Outcome'}

            if missing_cols_in_file:
                 print(f"  Warning: The following columns specified in 'relevant_columns_to_keep' are MISSING in the input file '{input_csv}' and will be skipped:")
                 print(f"  {sorted(list(missing_cols_in_file))}")
                 # If essential columns are missing, you might want to exit() here
                 # or ensure the first script generated them correctly.
            else:
                 print(f"  All requested columns found in {input_csv}.")

        except FileNotFoundError:
             print(f"Error: Input file for header check not found at {input_csv}")
             return False
        except Exception as e:
             print(f"Error reading header for {input_csv}: {e}")
             return False

        # Prepare the final list of columns including Outcome for writing
        final_output_columns = actual_cols_to_use_in_file + ['Outcome']


        reader = pd.read_csv(input_csv, chunksize=chunk_size, dtype=str, low_memory=False)

        for chunk_df in reader:
            chunk_count += 1
            current_rows = len(chunk_df)
            total_rows_processed += current_rows
            if chunk_count % 5 == 1 or chunk_size < 50000:
                print(f"  Processing chunk {chunk_count} ({current_rows} rows)...")

            # 1. Add outcome
            chunk_df['Outcome'] = outcome_val

            # 2. Filter columns based on what's ACTUALLY available IN THIS FILE
            # Make sure to include 'Outcome' which we just added
            chunk_df_filtered = chunk_df[final_output_columns].copy()

            # 3. Replace missing codes
            for code in missing_codes:
                chunk_df_filtered.replace(code, np.nan, inplace=True)

            # 4. Append to output CSV
            header_to_write = write_header and first_chunk_for_this_file
            chunk_df_filtered.to_csv(
                output_csv,
                mode='a',
                header=header_to_write, # Write header only if flag is True and it's the first chunk
                index=False,
                columns=final_output_columns, # Ensure consistent column order on write
                lineterminator='\n'
            )
            first_chunk_for_this_file = False

        print(f"Finished processing {input_csv}. Processed {total_rows_processed} rows.")
        return True

    except FileNotFoundError:
        print(f"Error: Input file not found during chunk processing: {input_csv}")
        return False
    except Exception as e:
        print(f"An error occurred while processing {input_csv}: {e}")
        return False

# --- Main Execution ---

if OUTPUT_DIR != '.' and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# Remove existing combined file to start fresh
if os.path.exists(combined_output_path):
    print(f"Removing existing output file: {combined_output_path}")
    os.remove(combined_output_path)

# 1. Process Natality Data (Write header for the first chunk)
print("\n--- Combining: Processing Natality Data (Live Births) ---")
success_natality = process_and_append_chunked(
    input_csv=natality_input_path,            # Use the LARGE processed natality CSV
    output_csv=combined_output_path,          # Write to the SINGLE combined output
    outcome_val=1,                            # Live birth
    cols_to_keep=relevant_columns_to_keep,    # Use the curated list
    missing_codes=missing_value_codes,
    chunk_size=CHUNK_SIZE,
    write_header=True                         # YES, write header for the first file
)

# 2. Process Fetal Death Data (Append, no header)
if success_natality:
    print("\n--- Combining: Processing Fetal Death Data (Stillbirths) ---")
    success_fetal = process_and_append_chunked(
        input_csv=fetal_death_input_path,         # Use the processed fetal death CSV
        output_csv=combined_output_path,          # Append to the SAME combined output
        outcome_val=0,                            # Stillbirth
        cols_to_keep=relevant_columns_to_keep,    # Use the SAME curated list
        missing_codes=missing_value_codes,
        chunk_size=CHUNK_SIZE,
        write_header=False                        # NO, do not write header again
    )
else:
    print("\nSkipping Fetal Death processing due to errors in Natality processing.")
    success_fetal = False

# --- Final Summary ---
if success_natality and success_fetal:
    print(f"\nSuccessfully created combined file: {combined_output_path}")
    print("This file should now contain both live births and stillbirths.")
    print("You can now use this file for ratio checking and model training.")
else:
    print("\nScript finished with errors during the combination process.")

print("\nCombination script finished.")