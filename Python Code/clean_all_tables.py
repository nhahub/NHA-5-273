from pathlib import Path
import numpy as np
import pandas as pd
# -----------------------------------------------------------------------------
# File paths and table order
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "Dataset_V5_EDA.xlsx"
OUTPUT_EXCEL = BASE_DIR / "Cleaned_Healthcare_Data.xlsx"
OUTPUT_CSV = BASE_DIR / "Cleaned_CSV_Files"
# All actual data tables in the source workbook
DATA_TABLES = [
    "Patients",
    "Encounters",
    "Hospitals",
    "Departments",
    "Hospital_Departments",
    "Appointments",
    "Lab_Tests",
    "Treatments",
    "ICU_Admissions",
    "Deaths",
    "Staff",
    "Daily_Hospital_Stats"
]
# -----------------------------------------------------------------------------
# General Shared Functions
# -----------------------------------------------------------------------------
def load_sheet_data(file_path, sheet_name):
    # Load one sheet and print its dimensions
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    print(f"Loaded '{sheet_name}': {data.shape[0]:,} rows, "f"{data.shape[1]:,} columns")
    return data

def strip_text_columns(df, columns=None, normalize_spaces=False):
    # Trim text columns
    if columns is None:
        columns = df.select_dtypes(include=["object", "string"]).columns

    for col in columns:
        if col in df.columns:
            values = df[col].astype("string")
            if normalize_spaces:
                values = values.str.replace(r"\s+", " ", regex=True)
            df[col] = values.str.strip()
    return df

def format_text_columns(df, columns):
    # Apply the original Title Case formatting to selected columns
    for col in columns:
        if col in df.columns:
            # Keep the original behavior: nulls in these fields become 'Nan'.
            df[col] = df[col].astype("string").str.strip().str.title()
    return df


def normalize_gender(df):
    # Standardize gender values to Male, Female, or Unknown
    def clean_value(value):
        if pd.isna(value):
            return "Unknown"
        value = str(value).strip().capitalize()
        if value in ["Male", "M"]:
            return "Male"
        if value in ["Female", "F"]:
            return "Female"
        return "Unknown"

    df["gender"] = df["gender"].apply(clean_value)
    return df


def handle_age_outliers(df):
    # Convert age to numeric, invalidate ages outside 0-120, and fill with median
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df.loc[(df["age"] < 0) | (df["age"] > 120), "age"] = np.nan
    df["age"] = df["age"].fillna(df["age"].median())
    return df


def normalize_dates(df, date_columns):
    # Convert general/mixed date columns to Python date values
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="mixed", errors="coerce").dt.date
    return df

def parse_known_date_formats(series):
    # Parse the project's mixed dates using its existing format rules:
    # YYYY... = ISO/year-first, slash = month-first, dash = day-first.
    values = series.astype("string").str.strip()
    is_year_first = values.str.match(r"^\d{4}", na=False)
    is_slash_date = values.str.match(r"^\d{2}/", na=False)
    is_dash_date = values.str.match(r"^\d{2}-", na=False)
    dates = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    if is_year_first.any():
        dates.loc[is_year_first] = pd.to_datetime(
            values.loc[is_year_first].str.replace("/", "-", regex=False),format="mixed",errors="coerce").dt.normalize()

    if is_slash_date.any():
        dates.loc[is_slash_date] = pd.to_datetime(
            values.loc[is_slash_date],format="%m/%d/%Y",errors="coerce").dt.normalize()

    if is_dash_date.any():
        dates.loc[is_dash_date] = pd.to_datetime(
            values.loc[is_dash_date],format="%d-%m-%Y",errors="coerce").dt.normalize()
    return dates

def normalize_mixed_format_dates(df, date_columns):
    # Use the shared project-specific parser on selected date columns
    for col in date_columns:
        if col in df.columns:
            df[col] = parse_known_date_formats(df[col])
    return df

def normalize_flag_column(df, col_name):
    # Standardize common boolean/flag values to Yes, No, Partially, Unknown
    def clean_flag_value(value):
        if pd.isna(value):
            return "Unknown"
        value = str(value).strip().lower()
        if value in ["true", "1", "yes", "y"]:
            return "Yes"
        if value in ["false", "0", "no", "n"]:
            return "No"
        if "partial" in value:
            return "Partially"
        return "Unknown"

    if col_name in df.columns:
        df[col_name] = df[col_name].apply(clean_flag_value)
    return df

def fix_date_logic(df):
    # Swap admission and discharge dates when discharge is earlier
    mask = df["admission_date"] > df["discharge_date"]
    if mask.sum() > 0:
        old_admission = df.loc[mask, "admission_date"].copy()
        df.loc[mask, "admission_date"] = df.loc[mask, "discharge_date"]
        df.loc[mask, "discharge_date"] = old_admission
    return df

def fill_missing_text(df):
    # Fill the original Encounters text fields with Unknown
    return df.fillna(
        {
            "diagnosis": "Unknown",
            "insurance_type": "Unknown",
            "outcome": "Unknown",
        }
    )

def round_float_columns(df):
    # Round float columns to one decimal place
    float_columns = df.select_dtypes(include="float").columns
    df[float_columns] = df[float_columns].round(1)
    return df
# -----------------------------------------------------------------------------
# Table-specific cleaning functions
# -----------------------------------------------------------------------------
def clean_patients(df):
    print("\nCleaning Patients table...")
    df = df.drop(columns=["source_record_count"], errors="ignore")
    df = normalize_gender(df)
    df = handle_age_outliers(df)
    df = format_text_columns(df, ["city", "country"])
    return df

def clean_encounters(df):
    print("\nCleaning Encounters table...")
    df = normalize_dates(df, ["admission_date", "discharge_date"])
    df = fix_date_logic(df)
    df = fill_missing_text(df)
    df = normalize_flag_column(df, "readmission_flag")
    return df

def get_hospital_type_from_name(name):
    # Infer a missing hospital type using the same rules as the original code
    name = str(name).lower()
    if "university" in name:
        return "Teaching"
    if "regional" in name:
        return "General"
    if "memorial" in name:
        return "Private"
    return "Public"

def clean_hospitals(df):
    print("\nCleaning Hospitals table...")
    # Handle Excel serial dates and text dates in the same column.
    numeric_dates = pd.to_numeric(df["established_date"], errors="coerce")
    dates_from_serial = pd.to_datetime(numeric_dates, unit="D", origin="1899-12-30", errors="coerce")
    dates_from_text = pd.to_datetime(
        df["established_date"].where(numeric_dates.isna()),
        errors="coerce",
        format="mixed")
    df["established_date"] = dates_from_serial.fillna(dates_from_text).dt.normalize().dt.date

    text_columns = [
        col for col in df.select_dtypes(include=["object", "string"]).columns
        if col != "established_date"
    ]
    df = strip_text_columns(df, text_columns)

    df["city"] = df["city"].str.title()
    df["hospital_type"] = df["hospital_type"].str.title()
    df["hospital_type"] = df["hospital_type"].replace({"Puplic": "Public"})

    # Keep the original referral-center mapping behavior.
    df["is_referral_center"] = df["is_referral_center"].astype(str).str.lower()
    df["is_referral_center"] = df["is_referral_center"].replace(
        {
            "true": "Yes",
            "yes": "Yes",
            "false": "No",
            "no": "No",
            "partially": "Partially",
            "nan": pd.NA,
        }
    )

    # Recalculate available beds, as in the original code.
    df["available_beds"] = df["total_beds"] - df["occupied_beds"]

    df["hospital_type"] = df["hospital_type"].fillna(
        df["hospital_name"].apply(get_hospital_type_from_name)
    )
    df["is_referral_center"] = df["is_referral_center"].fillna("Unknown")

    # Cap ratings at 5 and fill missing ratings with the post-cap average.
    df["rating"] = df["rating"].clip(upper=5.0)
    average_rating = df["rating"].mean()
    df["rating"] = df["rating"].fillna(average_rating)

    df = df.drop(columns=["source_record_count"], errors="ignore")
    df = round_float_columns(df)
    return df

def clean_appointments(df):
    print("\nCleaning Appointments table...")
    return normalize_dates(df, ["scheduled_date"])

def clean_lab_tests(df):
    print("\nCleaning Lab_Tests table...")
    # Repeated test_id values are retained because they represent different records.
    df = normalize_mixed_format_dates(df, ["test_date"])
    print("Lab date nulls:", df["test_date"].isnull().sum())
    df["test_date"] = df["test_date"].dt.date

    text_columns = [
        col for col in df.select_dtypes(
            include=["object", "string"]
        ).columns
        if col != "test_date"
    ]

    df = strip_text_columns(df, text_columns, normalize_spaces=True)

    # Partially and missing retest values were treated as No in the original code.
    df["is_retest"] = df["is_retest"].str.lower().replace(
        {
            "true": "Yes",
            "1": "Yes",
            "yes": "Yes",
            "false": "No",
            "0": "No",
            "no": "No",
            "partially": "No",
            "nan": "No",
            "none": "No",
        }
    )
    df["is_retest"] = df["is_retest"].fillna("No")

    df["turnaround_hours"] = df["turnaround_hours"].fillna(
        df.groupby("test_type")["turnaround_hours"].transform("mean")
    )
    # Remove the contradictory prefix and standardize missing reasons.
    df["reason_for_test"] = df["reason_for_test"].str.replace(
        r"^None,?\s*", "", regex=True
    )
    df["reason_for_test"] = df["reason_for_test"].str.capitalize()
    df["reason_for_test"] = df["reason_for_test"].replace("", pd.NA).fillna("Diagnostic")

    # Descriptive tests have no numeric result value or unit in this dataset.
    descriptive_tests = ["CT Scan", "MRI", "X-Ray", "ECG", "PCR"]
    is_descriptive = df["test_type"].isin(descriptive_tests)
    df.loc[is_descriptive, "test_value"] = 0
    df.loc[is_descriptive, "unit"] = np.nan

    # Fill missing numeric values with their test-type mean.
    numeric_tests = ["CBC", "Glucose", "Kidney Function"]
    for test_type in numeric_tests:
        mean_value = df.loc[df["test_type"] == test_type, "test_value"].mean()
        missing_value_mask = (
            (df["test_type"] == test_type) & df["test_value"].isnull()
        )
        df.loc[missing_value_mask, "test_value"] = mean_value

    # Fill missing units with the most common unit for each numeric test type.
    for test_type in numeric_tests:
        common_units = df.loc[df["test_type"] == test_type, "unit"].mode()
        if not common_units.empty:
            missing_unit_mask = (
                (df["test_type"] == test_type) & df["unit"].isnull()
            )
            df.loc[missing_unit_mask, "unit"] = common_units.iloc[0]

    df = round_float_columns(df)
    return df

def clean_treatments(df):
    print("\nCleaning Treatments table...")

    # Clean text using the shared function.
    treatment_text_columns = ["treatment_type", "side_effects", "administered_at"]
    df = strip_text_columns(df, treatment_text_columns)

    # Add treatment_date from the original date column using shared date logic.
    df["treatment_date"] = parse_known_date_formats(df["date"])

    print("Treatment date nulls:", df["treatment_date"].isnull().sum())
    print("Treatment date range:", df["treatment_date"].min(), "to", df["treatment_date"].max())

    # Keep is_followup numeric (1/0) so the existing statistical analysis still works.
    df["is_followup"] = df["is_followup"].astype(str).str.strip().str.lower()
    df["is_followup"] = df["is_followup"].map(
        {
            "1": 1,
            "1.0": 1,
            "true": 1,
            "yes": 1,
            "0": 0,
            "0.0": 0,
            "false": 0,
            "no": 0,
        }
    )

    # Remove the contradictory 'None, ' prefix and preserve missing values as Not recorded.
    df["side_effect"] = df["side_effects"].str.replace("None, ", "", regex=False)
    df["side_effect"] = df["side_effect"].fillna("Not recorded")

    # Dose status: procedures have no dose by design.
    procedures = ["Physiotherapy", "Surgery - Minor", "Surgery - Major"]

    def get_dose_status(row):
        if pd.notna(row["dose_number"]):
            return "Dose " + str(int(row["dose_number"]))
        if row["treatment_type"] in procedures:
            return "Not applicable"
        return "Unknown"

    df["dose_status"] = df.apply(get_dose_status, axis=1)

    # Extract the hour from the administration time.
    df["administered_hour"] = pd.to_datetime(
        df["administered_at"], format="%H:%M"
    ).dt.hour

    # Remove columns replaced or not needed by the original cleaning logic.
    df = df.drop(columns=["batch_number", "date", "side_effects"], errors="ignore")
    return df

def clean_icu_admissions(df):
    print("\nCleaning ICU_Admissions table...")
    df = strip_text_columns(df, ["survival_status", "discharged_to"])

    # Normalize both dates with the same parser used for Treatments and Lab_Tests.
    for original_col in ["icu_admission_date", "icu_discharge_date"]:
        parsed_dates = parse_known_date_formats(df[original_col])
        new_col = original_col.replace("_date", "_dt")
        df[new_col] = parsed_dates
        print(
            f"{original_col} -> nulls: {parsed_dates.isnull().sum()} | "
            f"range: {parsed_dates.min()} to {parsed_dates.max()}"
        )

    print(
        "Discharge before admission:",
        (df["icu_discharge_dt"] < df["icu_admission_dt"]).sum(),
    )

    # Length of stay and date keys for a later Date-table relationship.
    df["icu_los_days"] = (
        df["icu_discharge_dt"] - df["icu_admission_dt"]
    ).dt.days

    df = normalize_dates(df, ["icu_admission_dt", "icu_discharge_dt"])

    # Binary field used to calculate the ICU death rate.
    df["is_deceased"] = (df["survival_status"] == "Deceased").astype(int)

    # Remove columns specified in the original cleaning logic.
    df = df.drop(columns=["icu_admission_date","icu_discharge_date"],errors="ignore")
    return df

def clean_deaths(df):
    print("\nCleaning Deaths table...")
    df = format_text_columns(df, ["department"])
    df = normalize_dates(df, ["date_of_death"])
    df = normalize_flag_column(df, "autopsy_done")
    df = normalize_flag_column(df, "death_certificate_issued")

    # Remove "None" from contributing_factors.
    df["contributing_factors"] = (
        df["contributing_factors"].fillna("").apply(
            lambda value: ", ".join(
                item.strip()
                for item in str(value).split(",")
                if item.strip() and item.strip().lower() != "none")).replace("", pd.NA))
    return df

def clean_staff(df):
    print("\nCleaning Staff table...")
    df = normalize_dates(df, ["hire_date"])
    df = normalize_flag_column(df, "is_active")
    # Remove extra spaces from text columns.
    df = strip_text_columns(df,["department", "certification_status"],normalize_spaces=True)
    # Standardize department names.
    df["department"] = df["department"].str.title()
    # Replace missing certification statuses with Unknown.
    df["certification_status"] = (df["certification_status"].replace("", pd.NA).fillna("Unknown"))
    # Replace missing leave days with 0.
    df["leave_days"] = pd.to_numeric(df["leave_days"], errors="coerce").fillna(0)
    # Remove the unwanted column.
    df = df.drop(columns=["sick_leave_date"], errors="ignore")
    return df

def clean_daily_hospital_stats(df):
    print("\nCleaning Daily_Hospital_Stats table...")
    df = strip_text_columns(df,normalize_spaces=True)
    df["alert_level"] = df["alert_level"].str.capitalize()
    df = normalize_mixed_format_dates(df, ["date"])
    print("Stats date nulls:", df["date"].isnull().sum())
    df["date"] = df["date"].dt.date

    # Count duplicates excluding daily_stat_key; do not delete them.
    duplicate_count = df.duplicated(subset=[col for col in df.columns if col != "daily_stat_key"]).sum()
    print("Duplicates (kept, not deleted):", duplicate_count)
    df = round_float_columns(df)
    return df

# Only tables cleaned by the original script have cleaning functions here.
# Departments and Hospital_Departments are intentionally passed through unchanged.
TABLE_CLEANERS = {
    "Patients": clean_patients,
    "Encounters": clean_encounters,
    "Hospitals": clean_hospitals,
    "Appointments": clean_appointments,
    "Lab_Tests": clean_lab_tests,
    "Treatments": clean_treatments,
    "ICU_Admissions": clean_icu_admissions,
    "Deaths": clean_deaths,
    "Staff": clean_staff,
    "Daily_Hospital_Stats": clean_daily_hospital_stats,
}

def validate_input_file():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input workbook not found: {INPUT_FILE}")

    with pd.ExcelFile(INPUT_FILE) as workbook:
        missing_sheets = [
            sheet for sheet in DATA_TABLES
            if sheet not in workbook.sheet_names
        ]

    if missing_sheets:
        raise ValueError(f"These required sheets were not found: {missing_sheets}")

def clean_all_tables():
    cleaned_tables = {}
    for sheet_name in DATA_TABLES:
        df = load_sheet_data(INPUT_FILE, sheet_name)

        clean_function = TABLE_CLEANERS.get(sheet_name)

        if clean_function is None:
            print(f"Keeping '{sheet_name}' unchanged.")
        else:
            df = clean_function(df)

        cleaned_tables[sheet_name] = df
    return cleaned_tables

def export_to_excel(cleaned_tables):
    with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl", datetime_format="YYYY-MM-DD") as writer:
        for sheet_name, df in cleaned_tables.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def export_to_csv(cleaned_tables):
    # Create a folder to save each table as a separate CSV file
    output_dir = BASE_DIR / "Cleaned_CSV_Files"
    output_dir.mkdir(exist_ok=True)

    for sheet_name, df in cleaned_tables.items():
        file_path = output_dir / f"{sheet_name}.csv"
        df.to_csv(file_path, index=False, encoding="utf-8-sig")
        print(f"Exported individual CSV: {file_path}")
# -----------------------------------------------------------------------------
# Statistical diagnostics from the original Treatments and ICU code
# -----------------------------------------------------------------------------
def print_treatments_icu_analysis(df_trt, df_icu):
    # Print descriptive statistics for Treatments and ICU Admissions.
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("\n==================== Treatments: Statistical Analysis ====================")
    print(df_trt[["dose_number", "effectiveness_estimate", "is_followup", "administered_hour"]].describe().round(3))

    for col in ["treatment_type", "dose_status", "side_effect", "is_followup"]:
        counts = df_trt[col].value_counts(dropna=False)
        percentages = (counts / len(df_trt) * 100).round(1)
        print(pd.DataFrame({"count": counts,"percent %": percentages,}))

    print(
        df_trt.groupby("treatment_type")["effectiveness_estimate"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .round(3)
    )
    print((df_trt.groupby("treatment_type")["is_followup"].mean() * 100).round(1))
    print("Overall follow-up rate: %.1f%%" % (df_trt["is_followup"].mean() * 100))

    print("\n==================== ICU Admissions: Statistical Analysis ====================")
    icu_metrics = ["apache_score", "icu_bill_usd", "icu_los_days"]
    print(df_icu[icu_metrics].describe().round(2))
    print(
        pd.DataFrame(
            {
                "count": df_icu["survival_status"].value_counts(),
                "percent %": (
                    df_icu["survival_status"].value_counts(normalize=True) * 100
                ).round(1),
            }
        )
    )
    print("ICU death rate: %.1f%%" % (df_icu["is_deceased"].mean() * 100))
    print(df_icu.groupby("survival_status")[icu_metrics].mean().round(1))
    print(df_icu[icu_metrics].skew().round(2))
# -----------------------------------------------------------------------------
# Main workflow: clean once, then export once
# -----------------------------------------------------------------------------
def main():
    # 1. Check the input file and required sheets.
    validate_input_file()

    # 2. Load and clean all tables.
    cleaned_tables = clean_all_tables()

    # 3. Run statistical analysis.
    print_treatments_icu_analysis(
        cleaned_tables["Treatments"],
        cleaned_tables["ICU_Admissions"],
    )

    # 4. Export the results.
    export_to_excel(cleaned_tables)
    export_to_csv(cleaned_tables)

    # 5. Print the final summary.
    print("\n==================== Export complete ====================")
    print(f"Excel workbook: {OUTPUT_EXCEL}")
    print(f"CSV: {OUTPUT_CSV}")
    print("Tables exported:", ", ".join(cleaned_tables.keys()))
    print("CSV rows:", f"{sum(len(df) for df in cleaned_tables.values()):,}")

if __name__ == "__main__":
    main()