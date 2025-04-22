import pandas as pd
import io
import os

# --- Configuration ---
FETAL_DEATH_FILE_PATH = 'Take 2/1Unzippad/1.txt'
NATALITY_FILE_PATH = 'Take 2/RUSL/Rusl.txt'

# --- Output File Configuration ---
OUTPUT_DIR = '.'
FETAL_DEATH_CSV_FILENAME = 'Take 2/2csv/1.csv'
NATALITY_CSV_FILENAME = 'Take 2/RUSL/Rusl2.csv'

FETAL_DEATH_OUTPUT_PATH = os.path.join(OUTPUT_DIR, FETAL_DEATH_CSV_FILENAME)
NATALITY_OUTPUT_PATH = os.path.join(OUTPUT_DIR, NATALITY_CSV_FILENAME)

# --- Define Chunk Size ---
# Adjust this based on your available RAM. Start with 100,000 or 500,000.
# If you still get memory errors, make it smaller.
CHUNK_SIZE = 500000

# --- Column Specifications (Keep these long lists exactly as before) ---
# Fetal Death 2023 Layout
fetal_death_layout_spec = [
    ('VERSION', 7, 1),('OE_TABFLG', 10, 1),('DOD_YY', 11, 4),('DOD_MM', 15, 2),
    ('DOD_TT', 21, 4),('DOD_WK', 25, 1),('OSTATE', 26, 2),('OCNTYFIPS', 30, 3),
    ('OCNTYPOP', 33, 1),('BFACIL', 34, 1),('BFACIL3', 52, 1),('MAGE_IMPFLG', 84, 1),
    ('MAGE_REPFLG', 85, 1),('MAGER', 86, 2),('MAGER14', 88, 2),('MAGER9', 90, 1),
    ('MBCNTRY', 91, 2),('MBSTATE_REC', 97, 1),('MRSTATEPSTL', 102, 2),('MRCNTYFIPS', 104, 3),
    ('RCNTY_POP', 112, 1),('RECTYPE', 130, 1),('RESTATUS', 131, 1),('MRACE31', 132, 2),
    ('MRACE6', 134, 1),('MRACE15', 135, 2),('MRACEIMP', 138, 1),('UMHISP', 142, 1),
    ('MHISPX', 143, 1),('SR_MRACEHISP', 144, 1),('MEDUC', 145, 1),('FAGERPT_FLG', 172, 1),
    ('FAGECOMB', 177, 2),('FAGEREC11', 179, 2),('PRIORLIVE', 181, 2),('PRIORDEAD', 183, 2),
    ('LBO_REC', 187, 1),('ILLB_R', 197, 3),('ILLB_R11', 200, 2),('PRECARE', 202, 2),
    ('PRECARE_REC', 204, 1),('WIC', 229, 1),('CIG_0', 230, 2),('CIG_1', 232, 2),
    ('CIG_2', 234, 2),('CIG_3', 236, 2),('CIG_REC', 238, 1),('M_Ht_In', 243, 2),
    ('BMI', 245, 4),('BMI_R', 249, 1),('PWgt_R', 253, 3),('RF_DIAB', 257, 1),
    ('RF_GEST', 258, 1),('RF_PHYP', 259, 1),('RF_GHYP', 260, 1),('RF_ECLAM', 261, 1),
    ('RF_INFTR', 262, 1),('RF_FEDRG', 263, 1),('RF_ARTEC', 264, 1),('RF_CESAR', 265, 1),
    ('RF_CESARN', 266, 2),('ME_PRES', 274, 1),('ME_ROUT', 275, 1),('ME_TRIAL', 276, 1),
    ('RDMETH_REC', 277, 1),('DMETH_REC', 280, 1),('MM_RUPT', 281, 1),('MM_ICU', 282, 1),
    ('ATTEND', 283, 1),('DPLURAL', 301, 1),('IMP_PLUR', 303, 1),('SEX', 316, 1),
    ('IMP_SEX', 317, 1),('DLMP_MM', 318, 2),('DLMP_YY', 322, 4),('GEST_IMP', 329, 1),
    ('OBGEST_FLG', 330, 1),('COMBGEST', 331, 2),('GESTREC12', 333, 2),('GESTREC5', 335, 1),
    ('OEGest_Unedt', 336, 2),('COMBGEST_USED', 339, 1),('OEGest_Comb', 340, 2),('OEGest_R12', 342, 2),
    ('OEGest_R5', 344, 1),('DBWT', 349, 4),('BWTR14', 353, 2),('BWTR4', 355, 1),
    ('ESTOFD', 357, 1),('AUTOPSY', 358, 1),('HISTOPF', 359, 1),('AUTOPF', 360, 1),
    ('F_MEDUC', 372, 1),('F_CLINEST', 373, 1),('F_TOBACO', 374, 1),('F_M_HT', 375, 1),
    ('F_PWGT', 376, 1),('F_WIC', 377, 1),('F_RF_PDIAB', 378, 1),('F_RF_GDIAB', 379, 1),
    ('F_RF_PHYPER', 380, 1),('F_RF_GHYPER', 381, 1),('F_RF_ECLAMP', 382, 1),('F_RF_INFT', 383, 1),
    ('F_RF_INFT_DRG', 384, 1),('F_RF_INFT_ART', 385, 1),('F_RF_CESAR', 386, 1),('F_RF_NCESAR', 387, 1),
    ('F_MD_PRESENT', 388, 1),('F_MD_ROUTE', 389, 1),('F_MD_TRIAL', 390, 1),('F_MM_RUPTUR', 391, 1),
    ('F_MM_ICU', 392, 1),('F_MPCB', 403, 1),('F_CMBGST', 404, 1),('F_CIGS_0', 405, 1),
    ('F_CIGS_1', 406, 1),('F_CIGS_2', 407, 1),('F_CIGS_3', 408, 1),('F_FACILITY', 409, 1),
    ('DelMethRecF', 410, 1),('F_MORIGIN', 424, 1),('F_MRACE_R', 425, 1),('F_FAGE_u', 426, 1),
    ('F_DLLB_MM', 427, 1),('F_DLMP_MM', 429, 1),('F_DLMP_YY', 431, 1),('F_MAGE', 432, 1),
    ('F_ATTENDANT', 433, 1),('F_ESTOFD', 435, 1),('F_AUTOPSY', 436, 1),('F_HISTOPF', 437, 1),
    ('F_AUTOPF', 438, 1),('F_MRACEREC', 439, 1),('F_PLBL', 440, 1),('F_PLBD', 441, 1),
    ('F_DBWT', 442, 1),('F_BMI', 445, 1), ('IICOD', 2603, 5), ('IC_124_Fetal', 2643, 3),
    ('F_ICOD', 2651, 1)
]
natality_layout_spec = [
    ('DOB_YY', 9, 4),('DOB_MM', 13, 2),('DOB_TT', 19, 4),('DOB_WK', 23, 1),
    ('BFACIL', 32, 1),('F_BFACIL', 33, 1),('BFACIL3', 50, 1),('MAGE_IMPFLG', 73, 1),
    ('MAGE_REPFLG', 74, 1),('MAGER', 75, 2),('MAGER14', 77, 2),('MAGER9', 79, 1),
    ('MBSTATE_REC', 84, 1),('RESTATUS', 104, 1),('MRACE31', 105, 2),('MRACE6', 107, 1),
    ('MRACE15', 108, 2),('MRACEIMP', 111, 1),('MHISPX', 112, 1),('MHISP_R', 115, 1),
    ('F_MHISP', 116, 1),('MRACEHISP', 117, 1),('MAR_P', 119, 1),('DMAR', 120, 1),
    ('MAR_IMP', 121, 1),('F_MAR_P', 123, 1),('MEDUC', 124, 1),('F_MEDUC', 126, 1),
    ('FAGERPT_FLG', 142, 1),('FAGECOMB', 147, 2),('FAGEREC11', 149, 2),('FRACE31', 151, 2),
    ('FRACE6', 153, 1),('FRACE15', 154, 2),('FHISPX', 159, 1),('FHISP_R', 160, 1),
    ('F_FHISP', 161, 1),('FRACEHISP', 162, 1),('FEDUC', 163, 1),('f_FEDUC', 165, 1),
    ('PRIORLIVE', 171, 2),('PRIORDEAD', 173, 2),('PRIORTERM', 175, 2),('LBO_REC', 179, 1),
    ('TBO_REC', 182, 1),('ILLB_R', 198, 3),('ILLB_R11', 201, 2),('ILOP_R', 206, 3),
    ('ILOP_R11', 209, 2),('ILP_R', 214, 3),('ILP_R11', 217, 2),('PRECARE', 224, 2),
    ('F_MPCB', 226, 1),('PRECARE5', 227, 1),('PREVIS', 238, 2),('PREVIS_REC', 242, 2),
    ('F_TPCV', 244, 1),('WIC', 251, 1),('F_WIC', 252, 1),('CIG_0', 253, 2),
    ('CIG_1', 255, 2),('CIG_2', 257, 2),('CIG_3', 259, 2),('CIG0_R', 261, 1),
    ('CIG1_R', 262, 1),('CIG2_R', 263, 1),('CIG3_R', 264, 1),('F_CIGS_0', 265, 1),
    ('F_CIGS_1', 266, 1),('F_CIGS_2', 267, 1),('F_CIGS_3', 268, 1),('CIG_REC', 269, 1),
    ('F_TOBACO', 270, 1),('M_Ht_In', 280, 2),('F_M_HT', 282, 1),('BMI', 283, 4),
    ('BMI_R', 287, 1),('PWgt_R', 292, 3),('F_PWGT', 295, 1),('DWgt_R', 299, 3),
    ('F_DWGT', 303, 1),('WTGAIN', 304, 2),('WTGAIN_REC', 306, 1),('F_WTGAIN', 307, 1),
    ('RF_PDIAB', 313, 1),('RF_GDIAB', 314, 1),('RF_PHYPE', 315, 1),('RF_GHYPE', 316, 1),
    ('RF_EHYPE', 317, 1),('RF_PPTERM', 318, 1),('F_RF_PDIAB', 319, 1),('F_RF_GDIAB', 320, 1),
    ('F_RF_PHYPER', 321, 1),('F_RF_GHYPER', 322, 1),('F_RF_ECLAMP', 323, 1),('F_RF_PPB', 324, 1),
    ('RF_INFTR', 325, 1),('RF_FEDRG', 326, 1),('RF_ARTEC', 327, 1),('f_RF_INFT', 328, 1),
    ('F_RF_INF_DRG', 329, 1),('F_RF_INF_ART', 330, 1),('RF_CESAR', 331, 1),('RF_CESARN', 332, 2),
    ('F_RF_CESAR', 335, 1),('F_RF_NCESAR', 336, 1),('NO_RISKS', 337, 1),('IP_GON', 343, 1),
    ('IP_SYPH', 344, 1),('IP_CHLAM', 345, 1),('IP_HEPB', 346, 1),('IP_HEPC', 347, 1),
    ('F_IP_GONOR', 348, 1),('F_IP_SYPH', 349, 1),('F_IP_CHLAM', 350, 1),('F_IP_HEPATB', 351, 1),
    ('F_IP_HEPATC', 352, 1),('NO_INFEC', 353, 1),('OB_ECVS', 360, 1),('OB_ECVF', 361, 1),
    ('F_OB_SUCC', 363, 1),('F_OB_FAIL', 364, 1),('LD_INDL', 383, 1),('LD_AUGM', 384, 1),
    ('LD_STER', 385, 1),('LD_ANTB', 386, 1),('LD_CHOR', 387, 1),('LD_ANES', 388, 1),
    ('F_LD_INDL', 389, 1),('F_LD_AUGM', 390, 1),('F_LD_STER', 391, 1),('F_LD_ANTB', 392, 1),
    ('F_LD_CHOR', 393, 1),('F_LD_ANES', 394, 1),('NO_LBRDLV', 395, 1),('ME_PRES', 401, 1),
    ('ME_ROUT', 402, 1),('ME_TRIAL', 403, 1),('F_ME_PRES', 404, 1),('F_ME_ROUT', 405, 1),
    ('F_ME_TRIAL', 406, 1),('RDMETH_REC', 407, 1),('DMETH_REC', 408, 1),('F_DMETH_REC', 409, 1),
    ('MM_MTR', 415, 1),('MM_PLAC', 416, 1),('MM_RUPT', 417, 1),('MM_UHYST', 418, 1),
    ('MM_AICU', 419, 1),('F_MM_MTR', 421, 1),('F_MM_PLAC', 422, 1),('F_MM_RUPT', 423, 1),
    ('F_MM_UHYST', 424, 1),('F_MM_AICU', 425, 1),('NO_MMORB', 427, 1),('ATTEND', 433, 1),
    ('MTRAN', 434, 1),('PAY', 435, 1),('PAY_REC', 436, 1),('F_PAY', 437, 1),
    ('F_PAY_REC', 438, 1),('APGAR5', 444, 2),('APGAR5R', 446, 1),('F_APGAR5', 447, 1),
    ('APGAR10', 448, 2),('APGAR10R', 450, 1),('DPLURAL', 454, 1),('IMP_PLUR', 456, 1),
    ('SETORDER_R', 459, 1),('SEX', 475, 1),('IMP_SEX', 476, 1),('DLMP_MM', 477, 2),
    ('DLMP_YY', 481, 4),('COMPGST_IMP', 488, 1),('OBGEST_FLG', 489, 1),('COMBGEST', 490, 2),
    ('GESTREC10', 492, 2),('GESTREC3', 494, 1),('LMPUSED', 498, 1),('OEGest_Comb', 499, 2),
    ('OEGest_R10', 501, 2),('OEGest_R3', 503, 1),('DBWT', 504, 4),('BWTR12', 509, 2),
    ('BWTR4', 511, 1),('AB_AVEN1', 517, 1),('AB_AVEN6', 518, 1),('AB_NICU', 519, 1),
    ('AB_SURF', 520, 1),('AB_ANTI', 521, 1),('AB_SEIZ', 522, 1),('F_AB_VENT', 524, 1),
    ('F_AB_VENT6', 525, 1),('F_AB_NIUC', 526, 1),('F_AB_SURFAC', 527, 1),('F_AB_ANTIBIO', 528, 1),
    ('F_AB_SEIZ', 529, 1),('NO_ABNORM', 531, 1),('CA_ANEN', 537, 1),('CA_MNSB', 538, 1),
    ('CA_CCHD', 539, 1),('CA_CDH', 540, 1),('CA_OMPH', 541, 1),('CA_GAST', 542, 1),
    ('F_CA_ANEN', 543, 1),('F_CA_MENIN', 544, 1),('F_CA_HEART', 545, 1),('F_CA_HERNIA', 546, 1),
    ('F_CA_OMPHA', 547, 1),('F_CA_GASTRO', 548, 1),('CA_LIMB', 549, 1),('CA_CLEFT', 550, 1),
    ('CA_CLPAL', 551, 1),('CA_DOWN', 552, 1),('CA_DISOR', 553, 1),('CA_HYPO', 554, 1),
    ('F_CA_LIMB', 555, 1),('F_CA_CLEFTLP', 556, 1),('F_CA_CLEFT', 557, 1),('F_CA_DOWNS', 558, 1),
    ('F_CA_CHROM', 559, 1),('F_CA_HYPOS', 560, 1),('NO_CONGEN', 561, 1),('ITRAN', 567, 1),
    ('ILIVE', 568, 1),('BFED', 569, 1),('F_BFED', 570, 1)
]


# --- Helper function to generate pandas read_fwf colspecs ---
def generate_colspecs_and_names(layout_specification):
    """Generates colspecs and names for explicitly listed columns."""
    colspecs = []
    names = []
    for name, start_1based, length in layout_specification:
        start_0based = start_1based - 1
        end_0based = start_0based + length
        colspecs.append((start_0based, end_0based))
        names.append(name)
    return colspecs, names

# --- Parsing and Processing Function with Chunking ---
def process_fixed_width_file_chunked(input_path, output_path, layout_spec, chunk_size):
    """
    Parses a fixed-width file in chunks and saves to CSV.

    Args:
        input_path (str): Path to the input fixed-width data file.
        output_path (str): Path to save the output CSV file.
        layout_spec (list): List of tuples (col_name, start_pos_1_based, length).
        chunk_size (int): Number of rows per chunk.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    try:
        colspecs, names = generate_colspecs_and_names(layout_spec)
        print(f"Processing {input_path} -> {output_path} with {len(names)} defined columns...")
        print(f"Chunk size: {chunk_size} rows")

        first_chunk = True
        chunk_count = 0
        total_rows = 0

        # Create an iterator that yields DataFrames (chunks)
        reader = pd.read_fwf(
            input_path,
            colspecs=colspecs,
            names=names,
            header=None,
            dtype=str,
            encoding='latin-1',
            chunksize=chunk_size  # Enable chunking
        )

        # Open the output file only once
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            for chunk_df in reader:
                chunk_count += 1
                current_rows = len(chunk_df)
                total_rows += current_rows
                print(f"  Processing chunk {chunk_count} ({current_rows} rows)...")

                # Strip whitespace from string columns in the chunk
                for col in chunk_df.columns:
                    if chunk_df[col].dtype == 'object':
                        chunk_df[col] = chunk_df[col].str.strip()

                # --- Optional: Add data cleaning/type conversion per chunk here ---
                # Example: Convert specific columns known to be numeric
                # numeric_cols = ['DOD_YY', 'MAGER', 'DBWT'] # Adjust as needed
                # for col in numeric_cols:
                #     if col in chunk_df.columns:
                #         chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce')
                # ------------------------------------------------------------------

                # Write chunk to CSV
                # Write header only for the first chunk
                chunk_df.to_csv(f, header=first_chunk, index=False, lineterminator='\n')
                first_chunk = False # Don't write header for subsequent chunks

        print(f"\nSuccessfully processed {total_rows} rows in {chunk_count} chunks.")
        print(f"Output saved to: {output_path}")
        return True

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return False
    except Exception as e:
        print(f"An error occurred while processing {input_path}: {e}")
        return False

# --- Main Execution ---
print("--- Processing Fetal Death Data ---")
success_fetal = process_fixed_width_file_chunked(
    FETAL_DEATH_FILE_PATH,
    FETAL_DEATH_OUTPUT_PATH,
    fetal_death_layout_spec,
    CHUNK_SIZE
)

print("\n--- Processing Natality Data ---")
success_natality = process_fixed_width_file_chunked(
    NATALITY_FILE_PATH,
    NATALITY_OUTPUT_PATH,
    natality_layout_spec,
    CHUNK_SIZE
)


# --- Optional: Database Integration (Now easier with CSV) ---
# You can now load the generated CSVs into a database much more easily
# import sqlite3
# if success_fetal and success_natality:
#    try:
#        conn = sqlite3.connect('birth_data.db')
#        print("\nLoading processed Fetal Death CSV into database...")
#        df_fetal_processed = pd.read_csv(FETAL_DEATH_OUTPUT_PATH) # Read the CSV we just created
#        # Perform final type conversions/cleaning if needed before DB insert
#        df_fetal_processed.to_sql('fetal_deaths', conn, if_exists='replace', index=False)
#
#        print("Loading processed Natality CSV into database...")
#        df_natality_processed = pd.read_csv(NATALITY_OUTPUT_PATH)
#        # Perform final type conversions/cleaning if needed before DB insert
#        df_natality_processed.to_sql('natality', conn, if_exists='replace', index=False)
#
#        print("Data written to birth_data.db from CSVs")
#        conn.close()
#    except Exception as e:
#        print(f"Error writing to database from CSV: {e}")

print("\nScript finished.")