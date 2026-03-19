import os
import re
import pandas as pd
import numpy as np

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

BUT_CSV = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "PPG-BP Database",
    "brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0",
    "subject-info.csv"
)

OUTPUT_CSV = os.path.join(BACKEND_DIR, "data", "but_subject_dataset.csv")

LABELS = ["ht1", "ht2", "nt", "pt"]


def bp_to_label(sbp, dbp):
    try:
        sbp = float(sbp)
        dbp = float(dbp)
    except:
        return None

    if sbp >= 140 or dbp >= 90:
        return "ht2"
    elif 130 <= sbp <= 139 or 80 <= dbp <= 89:
        return "ht1"
    elif 120 <= sbp <= 129 and dbp < 80:
        return "pt"
    else:
        return "nt"


def parse_bp(bp_value):
    if pd.isna(bp_value):
        return None, None

    s = str(bp_value).strip()
    if not s:
        return None, None

    nums = re.findall(r"\d+", s)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])

    return None, None


def parse_motion(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()
    if not s:
        return np.nan

    nums = re.findall(r"\d+", s)
    if not nums:
        return np.nan

    return float(nums[0])


def main():
    if not os.path.isfile(BUT_CSV):
        print(f"ERROR: subject-info.csv not found:\n{BUT_CSV}")
        return

    df = pd.read_csv(BUT_CSV)

    if "Blood pressure [mmHg]" not in df.columns:
        print("ERROR: 'Blood pressure [mmHg]' column not found.")
        print("Available columns:", list(df.columns))
        return

    # Parse BP into SBP / DBP
    bp_parsed = df["Blood pressure [mmHg]"].apply(parse_bp)
    df["SBP"] = bp_parsed.apply(lambda x: x[0])
    df["DBP"] = bp_parsed.apply(lambda x: x[1])

    # Create label from SBP / DBP
    df["label"] = df.apply(lambda row: bp_to_label(row["SBP"], row["DBP"]), axis=1)

    # Encode gender
    if "Gender" in df.columns:
        df["Gender_num"] = df["Gender"].astype(str).str.upper().map({"M": 1, "F": 0})

    # Convert motion safely
    if "Motion" in df.columns:
        df["Motion_num"] = df["Motion"].apply(parse_motion)

    # IMPORTANT: Do NOT include SBP / DBP as features
    candidate_cols = [
        "Age [years]",
        "Height [cm]",
        "Weight [kg]",
        "Ear/finger",
        "Glycaemia [mmol/l]",
        "SpO2 [%]",
        "Gender_num",
        "Motion_num",
    ]

    usable_cols = [c for c in candidate_cols if c in df.columns]

    out = df[usable_cols + ["label"]].copy()

    # Keep only valid labels
    out = out[out["label"].isin(LABELS)]

    # Require some core non-BP fields if present
    required_core = [c for c in ["Age [years]", "Weight [kg]"] if c in out.columns]
    if required_core:
        out = out.dropna(subset=required_core)

    # Convert numeric columns and fill missing with median
    numeric_cols = [c for c in out.columns if c != "label"]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if not out[c].isna().all():
            out[c] = out[c].fillna(out[c].median())

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print("\n==================================================")
    print("  BUT SUBJECT DATASET CONVERTED SUCCESSFULLY")
    print("==================================================")
    print(f"  Input rows      : {len(df)}")
    print(f"  Usable rows     : {len(out)}")
    print(f"  Columns used    : {numeric_cols}")
    print(f"  Saved to        : {OUTPUT_CSV}")
    print("==================================================\n")

    print("Label distribution:")
    print(out["label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()