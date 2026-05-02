#!/usr/bin/env python3
"""
Create an Overleaf-ready LaTeX ablation table from LeMUQ/LARS ablation-study CSVs.

Expected folder layout, e.g.:
  ROOT_DIR/evqa_llava_bm25/*.csv
  ROOT_DIR/evqa_llava_eva_clip+contriever/*.csv
  ROOT_DIR/evqa_llava_rerank/*.csv
  ...

Expected ablation columns, e.g. for llava/evqa/eva_clip:
  LARS_BASE
  llava_evqa_eva_clip_base_no_context       -> LARS_Ret
  llava_evqa_eva_clip_base_with_context      -> LeMUQ
  llava_evqa_eva_clip_no_p_prime             -> LeMUQ - \tilde{p}'
  llava_evqa_eva_clip_no_p_q_i               -> LeMUQ - \tilde{p}_q^i
  llava_evqa_eva_clip_no_p_q_c               -> LeMUQ - \tilde{p}_q^c
  llava_evqa_eva_clip_no_p_q                 -> LeMUQ - \tilde{p}_q

Significance markers:
  \dag  : significant difference to LARS_Ret, DeLong test, p < alpha
  \ddag : significant difference to full LeMUQ, DeLong test, p < alpha
"""

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    from MLstatkit import Delong_test
except Exception:  # Allows the table to still be generated without markers.
    Delong_test = None


DEFAULT_ROOT_DIR = "experiment_results_ablation_study"
DEFAULT_OUTPUT_TEX = "ablation_table.tex"
DEFAULT_ALPHA = 0.05

DATASETS = ["evqa", "infoseek"]
MODELS = ["llava", "qwen"]
RETRIEVALS = ["bm25", "eva_clip", "rerank"]

DATASET_LABELS = {"evqa": "EVQA", "infoseek": "InfoSeek"}
MODEL_LABELS = {"llava": "LLaVA1.5-7B", "qwen": "Qwen3-VL-4B"}
RETRIEVAL_LABELS = {"bm25": "BM25", "eva_clip": "EVAC.", "rerank": "BM25+MLM"}

GEN_TO_TRAIN_RETRIEVAL = {
    "bm25": "bm25",
    "eva_clip+contriever": "eva_clip",
    "eva-clip+contriever": "eva_clip",
    "eva_clip_contriever": "eva_clip",
    "rerank": "rerank",
}

SKIP_COLS = {
    "question_text", "prediction", "ground_truths", "generated_text",
    "correctness", "context",
}

# key, displayed LaTeX label, column builder
ABLATION_ROWS = [
    ("accuracy", "Accuracy", lambda model, dataset, retr: "Accuracy"),
    ("lars_base", r"$\text{LARS}_{Base}$", lambda model, dataset, retr: "LARS_BASE"),
    ("lars_ret", r"$\text{LARS}_{Ret}$", lambda model, dataset, retr: f"{model}_{dataset}_{retr}_base_no_context"),
    ("lemuq", r"\textbf{LeMUQ}", lambda model, dataset, retr: f"{model}_{dataset}_{retr}_base_with_context"),
    ("no_p_prime", r"LeMUQ $-\tilde{p}'$", lambda model, dataset, retr: f"{model}_{dataset}_{retr}_no_p_prime"),
    ("no_p_q_i", r"LeMUQ $-\tilde{p}^{q,i}$", lambda model, dataset, retr: f"{model}_{dataset}_{retr}_no_p_q_i"),
    ("no_p_q_c", r"LeMUQ $-\tilde{p}^{q,c}$", lambda model, dataset, retr: f"{model}_{dataset}_{retr}_no_p_q_c"),
    ("no_p_q", r"LeMUQ $-\tilde{p}^{q}$", lambda model, dataset, retr: f"{model}_{dataset}_{retr}_no_p_q"),
]

EXCLUDE_FROM_BOLDING = {"Accuracy"}
SIGNIFICANCE_ROW_KEYS = {"lemuq", "no_p_prime", "no_p_q_i", "no_p_q_c", "no_p_q"}


def fmt3(x):
    if pd.isna(x):
        return "-"
    return f"{float(x):.3f}"


def robust_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    if len(y_true) == 0 or np.unique(y_true).size <= 1 or np.unique(y_score).size <= 1:
        return np.nan
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def normalize_generation_retrieval(name: str) -> str:
    return {
        "bm25": "bm25",
        "eva_clip+contriever": "eva_clip+contriever",
        "eva-clip+contriever": "eva_clip+contriever",
        "eva_clip_contriever": "eva_clip+contriever",
        "rerank": "rerank",
    }.get(str(name).strip(), str(name).strip())


def parse_folder_name(folder_name: str):
    m = re.match(r"^(evqa|infoseek)_(llava|qwen)_(.+)$", folder_name)
    if not m:
        return None
    dataset, model, gen_retrieval = m.groups()
    gen_retrieval = normalize_generation_retrieval(gen_retrieval)
    if gen_retrieval not in GEN_TO_TRAIN_RETRIEVAL:
        return None
    return dataset, model, GEN_TO_TRAIN_RETRIEVAL[gen_retrieval]


def load_and_concat_csvs(folder_path: str):
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        return None
    dfs = []
    for fp in csv_files:
        try:
            dfs.append(pd.read_csv(fp))
        except Exception as e:
            print(f"Could not read {fp}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None


def compute_experiment_metrics(df: pd.DataFrame):
    if df is None or df.empty or "correctness" not in df.columns:
        return None
    correctness = pd.to_numeric(df["correctness"], errors="coerce")
    if correctness.notna().sum() == 0:
        return None
    out = {"Accuracy": correctness.mean()}
    for col in df.columns:
        if col in SKIP_COLS:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        auc = robust_auc(correctness.values, vals.values)
        if not pd.isna(auc):
            out[col] = auc
    return out


def build_results_map(root_dir: str):
    results, raw_results, skipped = {}, {}, []
    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        parsed = parse_folder_name(folder_name)
        if parsed is None:
            skipped.append((folder_name, "unrecognized folder name"))
            continue
        dataset, model, retr = parsed
        df = load_and_concat_csvs(folder_path)
        metrics = compute_experiment_metrics(df)
        if metrics is None:
            skipped.append((folder_name, "no valid metrics"))
            continue
        key = (dataset, model, retr)
        results[key] = metrics
        raw_results[key] = df
    return results, raw_results, skipped


def get_metric(results, dataset, model, retr, metric_col):
    return results.get((dataset, model, retr), {}).get(metric_col, np.nan)


def clean_delong_inputs(df: pd.DataFrame, score_a_col: str, score_b_col: str):
    if df is None or df.empty or "correctness" not in df.columns:
        return None
    if score_a_col not in df.columns or score_b_col not in df.columns:
        return None
    y = pd.to_numeric(df["correctness"], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(df[score_a_col], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(df[score_b_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y, a, b = y[mask], a[mask], b[mask]
    if len(y) == 0 or np.unique(y).size <= 1 or not set(np.unique(y)).issubset({0.0, 1.0}):
        return None
    if np.unique(a).size <= 1 and np.unique(b).size <= 1:
        return None
    return y.astype(int), a.astype(float), b.astype(float)


def delong_significant(df: pd.DataFrame, reference_col: str, candidate_col: str, alpha: float):
    if Delong_test is None:
        return False
    cleaned = clean_delong_inputs(df, reference_col, candidate_col)
    if cleaned is None:
        return False
    y_true, ref_scores, cand_scores = cleaned
    if pd.isna(robust_auc(y_true, ref_scores)) or pd.isna(robust_auc(y_true, cand_scores)):
        return False
    try:
        _, p_value, *_ = Delong_test(
            y_true, ref_scores, cand_scores,
            alpha=0.95, return_ci=True, return_auc=True, verbose=0,
        )
        return bool(pd.notna(p_value) and float(p_value) < alpha)
    except Exception:
        return False


def compute_significance_maps(raw_results, alpha: float):
    """
    Key: (row_key, model, retr, dataset)
    dag_map  : row significantly differs from LARS_Ret
    ddag_map : row significantly differs from full LeMUQ
    """
    dag_map, ddag_map = {}, {}
    for model in MODELS:
        for retr in RETRIEVALS:
            for dataset in DATASETS:
                df = raw_results.get((dataset, model, retr))
                lars_ret_col = f"{model}_{dataset}_{retr}_base_no_context"
                lemuq_col = f"{model}_{dataset}_{retr}_base_with_context"
                for row_key, _, col_builder in ABLATION_ROWS:
                    if row_key not in SIGNIFICANCE_ROW_KEYS:
                        continue
                    candidate_col = col_builder(model, dataset, retr)
                    dag_map[(row_key, model, retr, dataset)] = delong_significant(
                        df, lars_ret_col, candidate_col, alpha
                    )
                    # Do not compare LeMUQ to itself for the double-dagger marker.
                    ddag_map[(row_key, model, retr, dataset)] = (
                        row_key != "lemuq" and delong_significant(df, lemuq_col, candidate_col, alpha)
                    )
    return dag_map, ddag_map


def extract_numeric_prefix(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    m = re.search(r"-?\d+\.\d+", str(x))
    return float(m.group(0)) if m else np.nan


def render_value(x):
    return x if isinstance(x, str) else fmt3(x)


def append_sig_superscripts(cell_content, add_dag=False, add_ddag=False):
    if cell_content == "-" or pd.isna(cell_content):
        return "-"
    suffix = ""
    if add_dag:
        suffix += r"\textsuperscript{\dag}"
    if add_ddag:
        suffix += r"\textsuperscript{\ddag}"
    return f"{cell_content}{suffix}"


def bold_best_per_column(rows, columns, exclude_labels=None):
    exclude_labels = exclude_labels or set()
    out_rows = [r.copy() for r in rows]
    for col in columns:
        candidates = []
        for i, row in enumerate(out_rows):
            if row["label"] in exclude_labels:
                continue
            val = extract_numeric_prefix(row.get(col, np.nan))
            if pd.notna(val):
                candidates.append((i, val))
        if not candidates:
            continue
        best_val = max(v for _, v in candidates)
        for i, v in candidates:
            if np.isclose(v, best_val):
                original = out_rows[i].get(col, np.nan)
                if isinstance(original, str):
                    m = re.search(r"(-?\d+\.\d+)", original)
                    if m:
                        num = m.group(1)
                        s, e = m.span(1)
                        out_rows[i][col] = original[:s] + rf"\textbf{{{num}}}" + original[e:]
                    else:
                        out_rows[i][col] = rf"\textbf{{{v:.3f}}}"
                else:
                    out_rows[i][col] = rf"\textbf{{{v:.3f}}}"
    return out_rows


def build_table_columns():
    cols = []
    for model in MODELS:
        for retr in RETRIEVALS:
            for dataset in DATASETS:
                cols.append((model, retr, dataset))
    return cols


def build_ablation_rows(results, dag_map, ddag_map):
    columns = build_table_columns()
    avg_col = "__AVG__"
    rows = []
    for row_key, display_label, col_builder in ABLATION_ROWS:
        row = {"key": row_key, "label": display_label}
        numeric_vals = []
        for model, retr, dataset in columns:
            metric_col = col_builder(model, dataset, retr)
            val = get_metric(results, dataset, model, retr, metric_col)
            cell = fmt3(val) if pd.notna(val) else "-"
            if row_key in SIGNIFICANCE_ROW_KEYS:
                cell = append_sig_superscripts(
                    cell,
                    add_dag=dag_map.get((row_key, model, retr, dataset), False),
                    add_ddag=ddag_map.get((row_key, model, retr, dataset), False),
                )
            row[(model, retr, dataset)] = cell
            if pd.notna(val):
                numeric_vals.append(float(val))
        row[avg_col] = fmt3(np.mean(numeric_vals)) if numeric_vals else "-"
        rows.append(row)
    return bold_best_per_column(rows, columns + [avg_col], exclude_labels=EXCLUDE_FROM_BOLDING)


def build_ablation_table_latex(results, dag_map, ddag_map):
    columns = build_table_columns()
    avg_col = "__AVG__"
    rows = build_ablation_rows(results, dag_map, ddag_map)

    lines = []
    lines.append(r"% === Ablation Study Table =====================")
    lines.append(r"% Requires in preamble:")
    lines.append(r"% \usepackage[table,dvipsnames]{xcolor}")
    lines.append(r"% \usepackage{multirow}")
    lines.append(r"% Optional if not already defined:")
    lines.append(r"% \newcommand{\shrink}{\resizebox{\columnwidth}{!}}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{2.9pt}")
    lines.append(
        r"\caption{Ablation study AUROC performance across EVQA and InfoSeek for LLaVA1.5-7B and Qwen3-VL-4B. "
        r"BM25, EVAC., and BM25+MLM are used as retrieval models. "
        r"$\text{LARS}_{Ret}$ uses the finetuned LARS model matched to the corresponding retrieval setting. "
        r"LeMUQ denotes the full method, and the remaining rows remove one component at a time. "
        r"Superscripts \textsuperscript{\dag} and \textsuperscript{\ddag} denote statistically significant differences according to the DeLong test ($p<0.05$), compared to $\text{LARS}_{Ret}$ and LeMUQ, respectively.}"
    )
    lines.append(r"\label{tab:lemuq_ablation}")
    lines.append(r"\shrink")
    lines.append(r"\begin{tabular}{l|ll|ll|ll|ll|ll|ll|l}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{LLM} & \multicolumn{6}{c|}{\textbf{LLaVA1.5-7B}} & "
        r"\multicolumn{6}{c|}{\textbf{Qwen3-VL-4B}} & \multirow{3}{*}{\textbf{Avg.}} \\ \cline{1-13}"
    )
    lines.append(
        r"\textbf{Retriever} & \multicolumn{2}{c}{\textbf{BM25}} & \multicolumn{2}{c}{\textbf{EVAC.}} & "
        r"\multicolumn{2}{c|}{\textbf{BM25+MLM}} & \multicolumn{2}{c}{\textbf{BM25}} & "
        r"\multicolumn{2}{c}{\textbf{EVAC.}} & \multicolumn{2}{c|}{\textbf{BM25+MLM}} & \\ \cline{1-13}"
    )
    lines.append(
        r"\textbf{} & EVQA & InfoSeek & EVQA & InfoSeek & EVQA & InfoSeek & "
        r"EVQA & InfoSeek & EVQA & InfoSeek & EVQA & InfoSeek & \\ \hline"
    )

    for row in rows:
        cells = [row["label"]]
        for col in columns:
            cells.append(render_value(row.get(col, "-")))
        cells.append(render_value(row.get(avg_col, "-")))
        line = " & ".join(cells) + r" \\"
        if row.get("key") == "accuracy":
            line = r"\rowcolor{gray!15} " + line
        lines.append(line)

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\miniskip")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def print_missing_columns_report(results):
    missing = []
    for dataset in DATASETS:
        for model in MODELS:
            for retr in RETRIEVALS:
                metrics = results.get((dataset, model, retr), {})
                for _, display_label, col_builder in ABLATION_ROWS:
                    metric_col = col_builder(model, dataset, retr)
                    if metric_col not in metrics:
                        missing.append((dataset, model, retr, display_label, metric_col))
    if missing:
        print("\nMissing metric columns or invalid AUROC columns:")
        for dataset, model, retr, display_label, metric_col in missing:
            print(f"  {dataset}/{model}/{retr}: {display_label} -> {metric_col}")


def main():
    parser = argparse.ArgumentParser(description="Build an Overleaf-ready LeMUQ ablation table.")
    parser.add_argument("--root-dir", default=DEFAULT_ROOT_DIR, help="Path to ablation-study result folders.")
    parser.add_argument("--output-tex", default=DEFAULT_OUTPUT_TEX, help="Output .tex filename.")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help="DeLong significance threshold.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir).expanduser().resolve()
    output_tex = Path(args.output_tex).expanduser().resolve()
    if not root_dir.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist: {root_dir}")

    results, raw_results, skipped = build_results_map(str(root_dir))
    expected = len(DATASETS) * len(MODELS) * len(RETRIEVALS)
    print(f"Found {len(results)}/{expected} ablation folders with readable results.")

    if skipped:
        print("\nSkipped folders:")
        for folder, reason in skipped:
            print(f"  {folder}: {reason}")
    if not results:
        raise ValueError("No ablation results found. Check --root-dir.")

    if Delong_test is None:
        print("\nWarning: MLstatkit.Delong_test could not be imported. Significance markers will be omitted.")

    print_missing_columns_report(results)
    dag_map, ddag_map = compute_significance_maps(raw_results, args.alpha)
    tex = build_ablation_table_latex(results, dag_map, ddag_map)
    output_tex.write_text(tex, encoding="utf-8")
    print(f"\nWrote LaTeX table to: {output_tex}")


if __name__ == "__main__":
    main()
