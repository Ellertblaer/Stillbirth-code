import pandas as pd
import numpy as np # For NaN values
import os

# --- Configuration ---
# Stóra settið
FETAL_DEATH_CSV_PATH = 'Stora_settid/fetal_deaths_2023_processed.csv'
NATALITY_CSV_PATH = 'Stora_settid/natality_2023_processed.csv'
COMBINED_OUTPUT_CSV_PATH = 'Stora_settid/combined_birth_data_2023_ml_ready_stora.csv'
# Litla settið
#FETAL_DEATH_CSV_PATH = 'Litla_settid//fetal_deaths_2023_processed.csv'
#NATALITY_CSV_PATH = 'Litla_settid//natality_2023_processed.csv'
#COMBINED_OUTPUT_CSV_PATH = 'combined_birth_data_2023_ml_ready.csv'

# Define where the input CSVs are and the output file name
OUTPUT_DIR = '.' # Assumes CSVs are in the same directory as the script

# Construct full paths
FETAL_DEATH_INPUT_PATH = os.path.join(OUTPUT_DIR, FETAL_DEATH_CSV_PATH)
NATALITY_INPUT_PATH = os.path.join(OUTPUT_DIR, NATALITY_CSV_PATH)
COMBINED_OUTPUT_PATH = os.path.join(OUTPUT_DIR, COMBINED_OUTPUT_CSV_PATH)


# --- Define Chunk Size ---
CHUNK_SIZE = 500000

# --- Column Selection ---
# !! REVISED LIST - BASED ON LIKELY NAMES IN PROCESSED CSVs !!
# !! VERIFY THIS AGAINST THE ACTUAL CSV HEADERS using the check script above !!
relevant_columns_to_keep = [
    # Timing
    'DOB_WK', # Use Natality name

    # Location/Facility
    'BFACIL',
    'BFACIL3',

    # Mother's Demographics
    'MAGER',
    'MBSTATE_REC',
    'RESTATUS',
    'MRACE31', # Assumed common detailed race
    'MHISPX', # Keep detailed Hispanic origin (check if in both CSVs)
    'MHISP_R', # Keep Hispanic Recode (check if in both CSVs)
    'MRACEHISP', # Keep combined Race/Hispanic (check if in both CSVs)
    'MEDUC',

    # Father's Demographics
    'FAGECOMB',
    # --- Add father's race/ethnicity if verified in both CSVs ---

    # Pregnancy History
    'PRIORLIVE',
    'PRIORDEAD',
    'LBO_REC',
    'ILLB_R',
    # --- Add Prior Terminations / Interval Last Preg if needed & verified ---

    # Prenatal Care
    'PRECARE', # Raw month number
    'PRECARE5', # Use Natality name for trimester recode
    'WIC',

    # Maternal Health / Habits
    'CIG_0', 'CIG_1', 'CIG_2', 'CIG_3', 'CIG_REC',
    'M_Ht_In', 'BMI', 'BMI_R', 'PWgt_R',
    # WTGAIN likely only in Natality, excluded for combining

    # Risk Factors (Assuming these names ARE consistent in the CSVs)
    # If the check script shows they are missing from natality_processed.csv,
    # you MUST fix the script that generated that file first.
    'RF_DIAB', 'RF_GEST', 'RF_PHYP', 'RF_GHYP',
    'RF_ECLAM', 'RF_INFTR', 'RF_FEDRG', 'RF_ARTEC', 'RF_CESAR', 'RF_CESARN',

    # Labor/Delivery Characteristics
    'ME_PRES', 'ME_ROUT', 'ME_TRIAL', 'RDMETH_REC', 'ATTEND', 'DPLURAL',

    # Infant Characteristics
    'SEX',
    'COMBGEST', # Combined gestation in weeks
    'OEGest_Comb',# Obstetric Estimate gestation in weeks
    'DBWT', # Birth Weight in grams

    # Target Variable
    'Outcome'
]

# Define missing value codes based on User Guides
missing_value_codes = {
    '9', '99', '999', '9999', 'U', 'X', ''
}

# --- Function to Process and Append CSV Chunks ---
def process_and_append_chunked(input_csv, output_csv, outcome_val, cols_to_keep, missing_codes, chunk_size, write_header):
    """
    Reads an input CSV in chunks, processes, and appends to an output CSV.
    """
    print(f"Processing {input_csv} -> {output_csv} (Outcome={outcome_val}, Header={write_header})")
    try:
        chunk_count = 0
        total_rows_processed = 0
        first_chunk_for_this_file = True
        all_expected_columns_present = True # Flag to track if all desired columns are found

        # Check header of the input file once before iterating
        try:
            input_cols = pd.read_csv(input_csv, nrows=0).columns.tolist() # Read only header
            # Find which requested columns are ACTUALLY in this specific input file
            actual_cols_to_use_in_file = [col for col in cols_to_keep if col in input_cols or col == 'Outcome']
            missing_cols_in_file = set(cols_to_keep) - set(actual_cols_to_use_in_file) - {'Outcome'} # Exclude Outcome here

            if missing_cols_in_file:
                 print(f"  Warning: The following columns from 'relevant_columns_to_keep' are MISSING in the input file '{input_csv}' and will be skipped for this file:")
                 print(f"  {sorted(list(missing_cols_in_file))}")
                 all_expected_columns_present = False # Some desired columns are missing
            else:
                 print(f"  All {len(actual_cols_to_use_in_file) -1} requested columns found in {input_csv}.") # -1 for Outcome

        except FileNotFoundError:
             print(f"Error: Input file for header check not found at {input_csv}")
             return False
        except Exception as e:
             print(f"Error reading header for {input_csv}: {e}")
             return False


        reader = pd.read_csv(input_csv, chunksize=chunk_size, dtype=str, low_memory=False)

        for chunk_df in reader:
            chunk_count += 1
            current_rows = len(chunk_df)
            total_rows_processed += current_rows
            # Limit excessive printing per chunk, maybe print every 10 chunks
            if chunk_count % 10 == 1 or chunk_size < 10000:
                 print(f"  Processing chunk {chunk_count} ({current_rows} rows)...")

            # 1. Add outcome
            chunk_df['Outcome'] = outcome_val

            # 2. Filter columns based on what's ACTUALLY available in this file
            #    We use the list determined from the header check above
            chunk_df_filtered = chunk_df[actual_cols_to_use_in_file].copy() # Use .copy() here to avoid SettingWithCopyWarning

            # 3. Replace missing codes
            for code in missing_codes:
                chunk_df_filtered.replace(code, np.nan, inplace=True) # Operate on the copy

            # 4. Append to output CSV
            header_to_write = write_header and first_chunk_for_this_file
            chunk_df_filtered.to_csv(
                output_csv,
                mode='a',
                header=header_to_write,
                index=False,
                lineterminator='\n'
            )
            first_chunk_for_this_file = False

        print(f"Finished processing {input_csv}. Processed {total_rows_processed} rows.")
        if not all_expected_columns_present:
             print(f"  Note: Some columns from 'relevant_columns_to_keep' were missing in {input_csv} as noted above.")
        return True

    except FileNotFoundError:
        print(f"Error: Input file not found during chunk processing at {input_csv}")
        return False
    except Exception as e:
        print(f"An error occurred while processing {input_csv}: {e}")
        return False

# --- Main Execution ---

if OUTPUT_DIR != '.' and not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

if os.path.exists(COMBINED_OUTPUT_PATH):
    print(f"Removing existing output file: {COMBINED_OUTPUT_PATH}")
    os.remove(COMBINED_OUTPUT_PATH)

print("\n--- Processing Natality Data (Live Births) ---")
success_natality = process_and_append_chunked(
    input_csv=NATALITY_INPUT_PATH,
    output_csv=COMBINED_OUTPUT_PATH,
    outcome_val=1,
    cols_to_keep=relevant_columns_to_keep,
    missing_codes=missing_value_codes,
    chunk_size=CHUNK_SIZE,
    write_header=True
)

if success_natality:
    print("\n--- Processing Fetal Death Data (Stillbirths) ---")
    success_fetal = process_and_append_chunked(
        input_csv=FETAL_DEATH_INPUT_PATH,
        output_csv=COMBINED_OUTPUT_PATH,
        outcome_val=0,
        cols_to_keep=relevant_columns_to_keep,
        missing_codes=missing_value_codes,
        chunk_size=CHUNK_SIZE,
        write_header=False
    )
else:
    print("\nSkipping Fetal Death processing due to errors in Natality processing.")
    success_fetal = False

if success_natality and success_fetal:
    print(f"\nSuccessfully created combined file: {COMBINED_OUTPUT_PATH}")
    print("Next steps: Load the combined CSV, perform final type conversions, handle remaining NaNs, and proceed with modeling.")
else:
    print("\nScript finished with errors during processing.")

print("\nScript finished.")

# --- Optional: Add database loading code here, reading from COMBINED_OUTPUT_PATH ---