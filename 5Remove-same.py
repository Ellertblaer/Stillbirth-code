import os

# --- Configuration ---
# File containing the lines you want to potentially remove rows from
FILE_TO_CLEAN = 'Take 2/4seinni-filter/summa.csv'

# File containing the lines that should be removed if found in FILE_TO_CLEAN
LINES_TO_REMOVE_FILE = 'Litla_settid/combined_birth_data_2023_ml_ready_litla-FINAL.csv'

# New file where the cleaned output will be saved
OUTPUT_FILE = 'Take 2/5remove-same/summa.csv'

# --- Optional: Configuration for Status Updates ---
REPORT_INTERVAL = 1000000 # Print status every N lines processed

print("--- Step 1: Loading lines to remove into memory ---")

lines_to_remove_set = set()
try:
    with open(LINES_TO_REMOVE_FILE, 'r', encoding='utf-8') as f_remove:
        for i, line in enumerate(f_remove):
            # Strip leading/trailing whitespace (including newline) for accurate matching
            lines_to_remove_set.add(line.strip())
            if (i + 1) % REPORT_INTERVAL == 0:
                 print(f"  Loaded {i + 1:,} lines to remove...")
    print(f"Finished loading. Found {len(lines_to_remove_set):,} unique lines to remove from '{LINES_TO_REMOVE_FILE}'.")

except FileNotFoundError:
    print(f"Error: File not found at '{LINES_TO_REMOVE_FILE}'. Please check the path.")
    exit()
except Exception as e:
    print(f"Error reading '{LINES_TO_REMOVE_FILE}': {e}")
    exit()

print(f"\n--- Step 2: Processing '{FILE_TO_CLEAN}' and writing unique lines to '{OUTPUT_FILE}' ---")

lines_read = 0
lines_written = 0
try:
    # Open input and output files safely
    with open(FILE_TO_CLEAN, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            lines_read += 1
            # Strip line from the main file for comparison
            stripped_line = line.strip()

            # Check if the stripped line is NOT in the set of lines to remove
            if stripped_line not in lines_to_remove_set:
                # If it's unique, write the ORIGINAL line (with newline) to the output
                f_out.write(line)
                lines_written += 1

            # Print progress update
            if lines_read % REPORT_INTERVAL == 0:
                print(f"  Processed {lines_read:,} lines, written {lines_written:,} unique lines...")

    print("\n--- Processing Complete ---")
    print(f"Total lines read from '{FILE_TO_CLEAN}': {lines_read:,}")
    print(f"Total unique lines written to '{OUTPUT_FILE}': {lines_written:,}")
    print(f"Number of lines removed: {(lines_read - lines_written):,}")

except FileNotFoundError:
    print(f"Error: File not found at '{FILE_TO_CLEAN}'. Please check the path.")
except Exception as e:
    print(f"Error processing '{FILE_TO_CLEAN}' or writing to '{OUTPUT_FILE}': {e}")

print("\nScript finished.")