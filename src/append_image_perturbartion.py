#!/usr/bin/env python3

import os
import pickle
import pandas as pd


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Run this script from the src folder.
SRC_DIR = os.path.abspath(os.getcwd())

# PKL result folders
PKL_ROOT = os.path.join(SRC_DIR, "results_mushtaq_large")

# CSV result folders
CSV_ROOT = os.path.join(SRC_DIR, "experiment_results_main")

# Column name
NEW_COLUMN_NAME = "Image_Perturbation"


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def load_l1_prob_values(pkl_path: str):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{pkl_path} does not contain a list. Found: {type(data)}")

    values = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(
                f"{pkl_path}: entry {i} is not a dict. Found: {type(entry)}"
            )
        if "l1_prob" not in entry:
            raise ValueError(f"{pkl_path}: entry {i} does not contain 'l1_prob'")
        values.append(entry["l1_prob"])

    return values


def append_column_to_csv(csv_path: str, values, column_name: str):
    df = pd.read_csv(csv_path)

    if len(df) != len(values):
        raise ValueError(
            f"Length mismatch for {csv_path}: "
            f"{len(df)} CSV rows vs {len(values)} PKL entries"
        )

    df[column_name] = values
    df.to_csv(csv_path, index=False)


def parse_csv_folder_name(csv_folder_name: str):
    """
    Example:
        evqa_llava_bm25 -> dataset='evqa', model='llava', method='bm25'
        infoseek_qwen_eva_clip+contriever -> dataset='infoseek', model='qwen', method='eva_clip+contriever'
    """
    parts = csv_folder_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"Unexpected CSV folder name format: {csv_folder_name}")

    dataset = parts[0]
    model = parts[1]
    method = "_".join(parts[2:])

    return dataset, model, method


def build_pkl_folder_name(dataset: str, model: str, method: str):
    """
    PKL folder format:
        evqa_bm25_llava
        evqa_doc+_qwen
        evqa_eva_clip+contriever_llava
        infoseek_None_qwen
    """
    return f"{dataset}_{method}_{model}"


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    print(f"Running from: {SRC_DIR}")
    print(f"PKL root: {PKL_ROOT}")
    print(f"CSV root: {CSV_ROOT}")
    print()

    if not os.path.isdir(PKL_ROOT):
        raise FileNotFoundError(f"PKL root folder not found: {PKL_ROOT}")

    if not os.path.isdir(CSV_ROOT):
        raise FileNotFoundError(f"CSV root folder not found: {CSV_ROOT}")

    csv_folders = sorted(
        folder for folder in os.listdir(CSV_ROOT)
        if os.path.isdir(os.path.join(CSV_ROOT, folder))
    )

    written = 0
    skipped = 0
    errors = 0

    for csv_folder in csv_folders:
        print("--------------------------------------------------")
        print(f"CSV folder : {csv_folder}")

        try:
            dataset, model, method = parse_csv_folder_name(csv_folder)
            pkl_folder = build_pkl_folder_name(dataset, model, method)

            pkl_path = os.path.join(PKL_ROOT, pkl_folder, "results_1.pkl")
            csv_path = os.path.join(CSV_ROOT, csv_folder, "results.csv")

            print(f"PKL folder : {pkl_folder}")
            print(f"PKL path   : {pkl_path}")
            print(f"CSV path   : {csv_path}")

            if not os.path.exists(pkl_path):
                print("-> SKIP: PKL file not found")
                skipped += 1
                continue

            if not os.path.exists(csv_path):
                print("-> SKIP: CSV file not found")
                skipped += 1
                continue

            l1_prob_values = load_l1_prob_values(pkl_path)
            append_column_to_csv(csv_path, l1_prob_values, NEW_COLUMN_NAME)

            print(f"-> OK: wrote column '{NEW_COLUMN_NAME}' with {len(l1_prob_values)} values")
            written += 1

        except Exception as e:
            print(f"-> ERROR: {e}")
            errors += 1

    print()
    print("Done.")
    print(f"Written : {written}")
    print(f"Skipped : {skipped}")
    print(f"Errors  : {errors}")


if __name__ == "__main__":
    main()
