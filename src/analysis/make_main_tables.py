import os
import re
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from MLstatkit import Delong_test

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = "experiment_results_main"
OUTPUT_TEX = "all_overleaf_tables.tex"

GEN_ORDER = ["None", "doc-", "bm25", "eva_clip+contriever", "rerank", "doc+"]
GEN_ORDER_GENERALIZATION = ["bm25", "eva_clip+contriever", "rerank"]

GEN_HEADER_MAP = {
    "None": r"$\text{Doc}^{\times}$",
    "doc-": r"$\text{Doc}^{-}$",
    "bm25": "BM25",
    "eva_clip+contriever": "EVAC.",
    "rerank": "BM25+MLM",
    "doc+": r"$\text{Doc}^{+}$",
}

DATASETS = ["evqa", "infoseek"]
MODELS = ["llava", "qwen"]

DATASET_LABELS = {
    "evqa": "EVQA",
    "infoseek": "InfoSeek",
}
MODEL_LABELS = {
    "llava": "LLaVA1.5-7B",
    "qwen": "Qwen3-VL-4B",
}

TRAIN_RETRIEVALS = ["bm25", "eva_clip", "rerank"]
TRAIN_RETRIEVAL_LABELS = {
    "bm25": "BM25",
    "eva_clip": "EVAC",
    "rerank": "BM25+MLM",
}

SKIP_COLS = {
    "question_text",
    "prediction",
    "ground_truths",
    "generated_text",
    "correctness",
    "context",
}

BASE_METHOD_ROWS = [
    ("Accuracy", "Accuracy"),
    ("PE", "Entropy"),
    ("P(True)", "PTrue_VLM"),
    ("Ecc. Unc.", "EccentricityUncertainty"),
    ("Img. Per.", "Image_Perturbation"),
    ("P(Relevant)", "PRelevant_VLM"),
    (r"LARS$_{Base}$", "LARS_BASE"),
]

SKIP_METHOD_METRICS = {"PRelevant_VLM"}
SKIP_METHOD_LABELS = {"P(Relevant)"}

ALL_TRAINING_SETUPS = [
    ("evqa", "llava"),
    ("evqa", "qwen"),
    ("infoseek", "llava"),
    ("infoseek", "qwen"),
]

DELONG_P_THRESHOLD = 0.05

# Fixed-width layout
# Table 3
TABLE3_LABEL_COL_WIDTH = "2.2cm"
TABLE3_DATA_COL_WIDTH = "1.05cm"

# Tables 4 and 5
TABLE45_LABEL_COL_WIDTH = "1.2cm"
TABLE45_DATA_COL_WIDTH = "1.1cm"

# ============================================================
# HELPERS
# ============================================================

def latex_escape(text: str) -> str:
    text = str(text)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt3(x):
    if pd.isna(x):
        return "-"
    return f"{float(x):.3f}"


def robust_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(y_true) == 0:
        return np.nan
    if np.unique(y_true).size <= 1:
        return np.nan
    if np.unique(y_score).size <= 1:
        return np.nan

    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def parse_folder_name(folder_name: str):
    m = re.match(r"^(evqa|infoseek)_(llava|qwen)_(.+)$", folder_name)
    if not m:
        return None
    dataset, model, retrieval = m.groups()
    return dataset, model, retrieval


def normalize_generation_retrieval(name: str) -> str:
    name = str(name).strip()
    mapping = {
        "None": "None",
        "none": "None",
        "doc-": "doc-",
        "bm25": "bm25",
        "eva_clip+contriever": "eva_clip+contriever",
        "eva-clip+contriever": "eva_clip+contriever",
        "eva_clip_contriever": "eva_clip+contriever",
        "rerank": "rerank",
        "doc+": "doc+",
    }
    return mapping.get(name, name)


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

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


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
    metrics_results = {}
    raw_results = {}

    for folder_name in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        parsed = parse_folder_name(folder_name)
        if parsed is None:
            continue

        dataset, model, gen_retrieval = parsed
        gen_retrieval = normalize_generation_retrieval(gen_retrieval)

        df = load_and_concat_csvs(folder_path)
        metrics = compute_experiment_metrics(df)

        if metrics is None:
            print(f"Skipping {folder_name}: no valid metrics.")
            continue

        key = (dataset, model, gen_retrieval)
        metrics_results[key] = metrics
        raw_results[key] = df

    return metrics_results, raw_results


def train_col(model: str, dataset: str, retrieval: str, with_context: bool = False) -> str:
    col = f"{model}_{dataset}_{retrieval}"
    if with_context:
        col += "_with_context"
    return col


def get_metric(results, dataset, model, gen_retrieval, metric_col):
    return results.get((dataset, model, gen_retrieval), {}).get(metric_col, np.nan)


def get_table_columns():
    cols = []
    for dataset in DATASETS:
        for gen_ret in GEN_ORDER:
            cols.append((dataset, gen_ret))
    return cols


def get_enabled_base_method_rows():
    return [
        (display_name, metric_col)
        for display_name, metric_col in BASE_METHOD_ROWS
        if metric_col not in SKIP_METHOD_METRICS and display_name not in SKIP_METHOD_LABELS
    ]


def is_bold_string(x):
    return isinstance(x, str) and x.startswith(r"\textbf{")


def extract_numeric_prefix(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x)
    m = re.search(r"-?\d+\.\d+", s)
    if m:
        return float(m.group(0))

    return np.nan


def render_value(x):
    if is_bold_string(x):
        return x
    if isinstance(x, str):
        return x
    return fmt3(x)


def bold_best_per_column(rows, columns, exclude_labels=None):
    exclude_labels = exclude_labels or set()
    out_rows = [r.copy() for r in rows]

    model_blocks = []
    current_start = None
    current_name = None

    for i, row in enumerate(out_rows):
        if row.get("is_model_section", False):
            if current_start is not None:
                model_blocks.append((current_name, current_start, i))
            current_name = row["label"]
            current_start = i + 1

    if current_start is not None:
        model_blocks.append((current_name, current_start, len(out_rows)))

    if not model_blocks:
        model_blocks = [("__all__", 0, len(out_rows))]

    for _, start, end in model_blocks:
        for col in columns:
            candidates = []

            for i in range(start, end):
                row = out_rows[i]

                if row.get("is_model_section", False) or row.get("is_section", False):
                    continue

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
                            start_idx, end_idx = m.span(1)
                            out_rows[i][col] = (
                                original[:start_idx]
                                + rf"\textbf{{{num}}}"
                                + original[end_idx:]
                            )
                        else:
                            out_rows[i][col] = rf"\textbf{{{v:.3f}}}"
                    else:
                        out_rows[i][col] = rf"\textbf{{{v:.3f}}}"

    return out_rows


def build_2dataset_header():
    lines = []
    lines.append(
        r"& \multicolumn{6}{c}{\textbf{EVQA}} & \multicolumn{6}{c}{\textbf{InfoSeek}} \\ \hline"
    )

    header = [r"\textbf{UE Method}"]
    for dataset in DATASETS:
        for g in GEN_ORDER:
            header.append(rf"\textbf{{{get_gen_header(dataset, g)}}}")

    lines.append(" & ".join(header) + r" \\ \hline")
    return lines


def get_gen_header(dataset: str, gen_ret: str) -> str:
    if dataset == "infoseek" and gen_ret == "doc+":
        return r"$\text{Doc}^{+*}$"
    return GEN_HEADER_MAP[gen_ret]


def gen_ret_from_train_retrieval(train_retrieval: str) -> str:
    mapping = {
        "bm25": "bm25",
        "eva_clip": "eva_clip+contriever",
        "rerank": "rerank",
    }
    return mapping[train_retrieval]


def format_ood_method_label(kind: str, retr: str | None = None):
    if retr is None:
        return r"LARS$_{Base}$"

    retr_label = TRAIN_RETRIEVAL_LABELS[retr]

    if retr == "rerank":
        if kind == "lars":
            return r"\shortstack[c]{LARS$_{\mathrm{BM25{+}MLM}}$}"
        return r"\shortstack[c]{MMRA$_{\mathrm{BM25{+}MLM}}$}"

    if kind == "lars":
        return rf"LARS$_{{{retr_label}}}$"
    return rf"MMRA$_{{{retr_label}}}$"


def format_compact_setup_label(kind: str):
    if kind == "lars":
        return r"$\text{LARS}_{Ret.}$"
    return r"\shortstack[c]{\textbf{$\text{MMRA}$}}"


def format_two_line_header(label: str) -> str:
    if label == "BM25+MLM":
        return r"\shortstack[c]{\textbf{BM25+}\\\textbf{MLM}}"
    return rf"\textbf{{{label}}}"


def build_fixed_width_colspec(n_data_cols: int, label_width: str, data_width: str) -> str:
    return (
        rf">{{\raggedright\arraybackslash}}m{{{label_width}}}|"
        + "".join(
            rf">{{\centering\arraybackslash}}m{{{data_width}}}"
            + ("|" if i == (n_data_cols // 2 - 1) else "")
            for i in range(n_data_cols)
        )
    )


def append_sig_superscripts(cell_content, add_dag=False, add_ddag=False):
    if cell_content == "-" or pd.isna(cell_content):
        return "-"

    suffix = ""
    if add_dag:
        suffix += r"\textsuperscript{\dag}"
    if add_ddag:
        suffix += r"\textsuperscript{\ddag}"
    return f"{cell_content}{suffix}"


# ============================================================
# DELONG HELPERS
# ============================================================

def get_clean_binary_auc_inputs(df: pd.DataFrame, y_col: str, score_a_col: str, score_b_col: str):
    if df is None or df.empty:
        return None

    required = [y_col, score_a_col, score_b_col]
    for col in required:
        if col not in df.columns:
            return None

    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(df[score_a_col], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(df[score_b_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(y) & np.isfinite(a) & np.isfinite(b)
    y = y[mask]
    a = a[mask]
    b = b[mask]

    if len(y) == 0:
        return None

    unique_y = np.unique(y)
    if unique_y.size <= 1:
        return None

    if not set(unique_y).issubset({0.0, 1.0}):
        return None

    if np.unique(a).size <= 1 and np.unique(b).size <= 1:
        return None
    return y.astype(int), a.astype(float), b.astype(float)


def compute_delong_for_pair(df: pd.DataFrame, score_a_col: str, score_b_col: str):
    cleaned = get_clean_binary_auc_inputs(df, "correctness", score_a_col, score_b_col)
    if cleaned is None:
        return None

    y_true, scores_a, scores_b = cleaned

    n_total = len(y_true)
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    auc_a = robust_auc(y_true, scores_a)
    auc_b = robust_auc(y_true, scores_b)

    if pd.isna(auc_a) or pd.isna(auc_b):
        return None

    try:
        z_score, p_value, ci_a, ci_b, auc_a_dl, auc_b_dl, info = Delong_test(
            y_true,
            scores_a,
            scores_b,
            alpha=0.95,
            return_ci=True,
            return_auc=True,
            verbose=0,
        )

        method = info.get("method", "unknown") if isinstance(info, dict) else "unknown"
        var_diff = info.get("var_diff", np.nan) if isinstance(info, dict) else np.nan

        if auc_a_dl is not None and not pd.isna(auc_a_dl):
            auc_a = float(auc_a_dl)
        if auc_b_dl is not None and not pd.isna(auc_b_dl):
            auc_b = float(auc_b_dl)

        return {
            "n": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auc_a": float(auc_a),
            "auc_b": float(auc_b),
            "diff_b_minus_a": float(auc_b - auc_a),
            "z_score": float(z_score) if z_score is not None else np.nan,
            "p_value": float(p_value) if p_value is not None else np.nan,
            "ci_a_low": ci_a[0] if ci_a is not None else np.nan,
            "ci_a_high": ci_a[1] if ci_a is not None else np.nan,
            "ci_b_low": ci_b[0] if ci_b is not None else np.nan,
            "ci_b_high": ci_b[1] if ci_b is not None else np.nan,
            "method": method,
            "var_diff": var_diff,
            "significant": bool(pd.notna(p_value) and float(p_value) < DELONG_P_THRESHOLD),
        }
    except Exception as e:
        return {
            "n": n_total,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "auc_a": float(auc_a),
            "auc_b": float(auc_b),
            "diff_b_minus_a": float(auc_b - auc_a),
            "z_score": np.nan,
            "p_value": np.nan,
            "ci_a_low": np.nan,
            "ci_a_high": np.nan,
            "ci_b_low": np.nan,
            "ci_b_high": np.nan,
            "method": f"error:{type(e).__name__}",
            "var_diff": np.nan,
            "significant": False,
        }


# ============================================================
# SIGNIFICANCE MAPS
# ============================================================

def compute_within_distribution_significance(raw_results):
    """
    Returns:
      dag_map : MMRA vs LARS_BASE
      ddag_map: MMRA vs matched LARS_Ret.
      summary_df: summary for matched LARS_Ret. vs MMRA
    """
    dag_map = {}
    ddag_map = {}
    summary_rows = []

    for eval_model in MODELS:
        for eval_dataset in DATASETS:
            for gen_ret in GEN_ORDER_GENERALIZATION:
                df = raw_results.get((eval_dataset, eval_model, gen_ret))
                if df is None:
                    continue

                retr = {
                    "bm25": "bm25",
                    "eva_clip+contriever": "eva_clip",
                    "rerank": "rerank",
                }[gen_ret]

                lars_base_col = "LARS_BASE"
                lars_ret_col = train_col(eval_model, eval_dataset, retr, with_context=False)
                mmra_col = train_col(eval_model, eval_dataset, retr, with_context=True)

                base_vs_mmra = compute_delong_for_pair(df, lars_base_col, mmra_col)
                lars_vs_mmra = compute_delong_for_pair(df, lars_ret_col, mmra_col)

                key = (eval_model, eval_dataset, retr, gen_ret)

                dag_map[key] = bool(base_vs_mmra is not None and base_vs_mmra["significant"])
                ddag_map[key] = bool(lars_vs_mmra is not None and lars_vs_mmra["significant"])

                if lars_vs_mmra is not None:
                    summary_rows.append({
                        "eval_model": eval_model,
                        "eval_model_label": MODEL_LABELS[eval_model],
                        "eval_dataset": eval_dataset,
                        "eval_dataset_label": DATASET_LABELS[eval_dataset],
                        "gen_retrieval": gen_ret,
                        "finetune_retrieval": retr,
                        "finetune_retrieval_label": TRAIN_RETRIEVAL_LABELS[retr],
                        "n": lars_vs_mmra["n"],
                        "n_pos": lars_vs_mmra["n_pos"],
                        "n_neg": lars_vs_mmra["n_neg"],
                        "auc_lars": lars_vs_mmra["auc_a"],
                        "auc_mmra": lars_vs_mmra["auc_b"],
                        "diff_mmra_minus_lars": lars_vs_mmra["diff_b_minus_a"],
                        "z_score": lars_vs_mmra["z_score"],
                        "p_value": lars_vs_mmra["p_value"],
                        "significant": lars_vs_mmra["significant"],
                        "method": lars_vs_mmra["method"],
                        "var_diff": lars_vs_mmra["var_diff"],
                    })

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["eval_model", "finetune_retrieval", "eval_dataset", "gen_retrieval"]
        ).reset_index(drop=True)

    return dag_map, ddag_map, summary_df


def compute_ood_retriever_significance(raw_results):
    """
    Returns:
      dag_map : MMRA vs LARS_BASE
      ddag_map: MMRA vs corresponding LARS row
    Key: (model, dataset, retr, gen_ret)
    """
    dag_map = {}
    ddag_map = {}

    for model in MODELS:
        for dataset in DATASETS:
            for gen_ret in GEN_ORDER:
                df = raw_results.get((dataset, model, gen_ret))
                if df is None:
                    continue

                for retr in TRAIN_RETRIEVALS:
                    lars_base_col = "LARS_BASE"
                    lars_col = train_col(model, dataset, retr, with_context=False)
                    mmra_col = train_col(model, dataset, retr, with_context=True)

                    base_vs_mmra = compute_delong_for_pair(df, lars_base_col, mmra_col)
                    lars_vs_mmra = compute_delong_for_pair(df, lars_col, mmra_col)

                    key = (model, dataset, retr, gen_ret)
                    dag_map[key] = bool(base_vs_mmra is not None and base_vs_mmra["significant"])
                    ddag_map[key] = bool(lars_vs_mmra is not None and lars_vs_mmra["significant"])

    return dag_map, ddag_map


def compute_ood_dataset_significance(raw_results):
    """
    Compare LARS_Ret. vs MMRA for Table 4.
    Key: (model, src_dataset, tgt_dataset, retr)
    """
    ddag_map = {}

    for model in MODELS:
        for src_dataset, tgt_dataset in [("evqa", "infoseek"), ("infoseek", "evqa")]:
            for retr in TRAIN_RETRIEVALS:
                gen_ret = gen_ret_from_train_retrieval(retr)
                df = raw_results.get((tgt_dataset, model, gen_ret))
                if df is None:
                    continue

                lars_col = train_col(model, src_dataset, retr, with_context=False)
                mmra_col = train_col(model, src_dataset, retr, with_context=True)

                stats = compute_delong_for_pair(df, lars_col, mmra_col)
                key = (model, src_dataset, tgt_dataset, retr)
                ddag_map[key] = bool(stats is not None and stats["significant"])

    return ddag_map


def compute_ood_vlm_significance(raw_results):
    """
    Compare LARS_Ret. vs MMRA for Table 5.
    Key: (dataset, src_model, tgt_model, retr)
    """
    ddag_map = {}

    for dataset in DATASETS:
        for src_model, tgt_model in [("llava", "qwen"), ("qwen", "llava")]:
            for retr in TRAIN_RETRIEVALS:
                gen_ret = gen_ret_from_train_retrieval(retr)
                df = raw_results.get((dataset, tgt_model, gen_ret))
                if df is None:
                    continue

                lars_col = train_col(src_model, dataset, retr, with_context=False)
                mmra_col = train_col(src_model, dataset, retr, with_context=True)

                stats = compute_delong_for_pair(df, lars_col, mmra_col)
                key = (dataset, src_model, tgt_model, retr)
                ddag_map[key] = bool(stats is not None and stats["significant"])

    return ddag_map


# ============================================================
# TABLE 1: BIG ALL-RESULTS TABLE
# ============================================================

def build_big_table_rows_for_model(results, eval_model):
    rows = []

    for display_name, metric_col in get_enabled_base_method_rows():
        row = {"label": display_name, "is_section": False}
        for dataset, gen_ret in get_table_columns():
            v = get_metric(results, dataset, eval_model, gen_ret, metric_col)
            if metric_col == "PRelevant_VLM" and gen_ret == "None":
                v = np.nan
            row[(dataset, gen_ret)] = v
        rows.append(row)

    for train_dataset, train_model in ALL_TRAINING_SETUPS:
        rows.append({
            "label": f"__SECTION__:{DATASET_LABELS[train_dataset]} / {MODEL_LABELS[train_model]}",
            "is_section": True
        })

        for retr in TRAIN_RETRIEVALS:
            lars_row = {
                "label": rf"LARS$_{{{TRAIN_RETRIEVAL_LABELS[retr]}}}$",
                "is_section": False
            }
            mmra_row = {
                "label": rf"MMRA-LARS$_{{{TRAIN_RETRIEVAL_LABELS[retr]}}}$",
                "is_section": False
            }

            lars_col = train_col(train_model, train_dataset, retr, with_context=False)
            mmra_col = train_col(train_model, train_dataset, retr, with_context=True)

            for dataset, gen_ret in get_table_columns():
                lars_row[(dataset, gen_ret)] = get_metric(results, dataset, eval_model, gen_ret, lars_col)
                mmra_row[(dataset, gen_ret)] = get_metric(results, dataset, eval_model, gen_ret, mmra_col)

            rows.append(lars_row)
            rows.append(mmra_row)

    return rows


def build_big_table_latex(results):
    columns = get_table_columns()

    llava_rows = build_big_table_rows_for_model(results, "llava")
    qwen_rows = build_big_table_rows_for_model(results, "qwen")

    all_rows = []
    all_rows.append({"label": "__MODEL_SECTION__:LLaVA1.5-7B", "is_model_section": True})
    all_rows.extend(llava_rows)
    all_rows.append({"label": "__MODEL_SECTION__:Qwen3-VL-4B", "is_model_section": True})
    all_rows.extend(qwen_rows)

    all_rows = bold_best_per_column(
        all_rows,
        columns=columns,
        exclude_labels={"Accuracy"}
    )

    lines = []
    lines.append(r"% === Table 1: UQ performance")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\renewcommand{\arraystretch}{0.8}")
    lines.append(r"\setlength{\tabcolsep}{1pt}")
    lines.append(r"\begin{tabular}{l|cccccc|cccccc}")
    lines.append(r"\hline")
    lines.extend(build_2dataset_header())

    for row in all_rows:
        if row.get("is_model_section", False):
            section_name = row["label"].replace("__MODEL_SECTION__:", "")
            lines.append(
                rf"\rowcolor{{gray!20}}\multicolumn{{13}}{{l}}{{\textit{{{latex_escape(section_name)}}}}} \\"
            )
            continue

        if row.get("is_section", False):
            section_name = row["label"].replace("__SECTION__:", "")
            lines.append(
                rf"\multicolumn{{13}}{{l}}{{\hspace{{0.3em}}\textbf{{{latex_escape(section_name)}}}}} \\"
            )
            continue

        cells = [row["label"]]
        for col in columns:
            cells.append(render_value(row.get(col, np.nan)))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{AUROC performance of baseline UE methods, finetuned LARS, and MMRA-LARS across EVQA and InfoSeek for LLaVA1.5-7B and Qwen3-VL-4B. For each experiment column, the best-performing method is highlighted in bold.}"
    )
    lines.append(r"\label{tab:uq_performance_all}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ============================================================
# TABLE 2: WITHIN-DISTRIBUTION TABLE
# ============================================================

def get_within_table_columns():
    cols = []
    for model in MODELS:
        for gen_ret in GEN_ORDER_GENERALIZATION:
            for dataset in DATASETS:
                cols.append((model, gen_ret, dataset))
    return cols


def get_within_metric(results, eval_dataset, eval_model, gen_retrieval, metric_col):
    return get_metric(results, eval_dataset, eval_model, gen_retrieval, metric_col)


def format_mmra_gain_only(mmra_v, lars_v):
    if pd.isna(mmra_v) or pd.isna(lars_v):
        return "-"

    delta = float(mmra_v) - float(lars_v)
    if delta > 0:
        return rf"\scriptsize \textcolor{{ForestGreen}}{{(+{delta:.3f})}}"
    elif delta < 0:
        return rf"\scriptsize \textcolor{{BrickRed}}{{({delta:.3f})}}"
    else:
        return rf"\scriptsize \textcolor{{black}}{{(+0.000)}}"


def build_within_distribution_table_latex(results, dag_map, ddag_map):
    columns = get_within_table_columns()
    avg_col = "__AVG__"

    single_rows = []

    within_baseline_rows = [
        ("Accuracy", "Accuracy"),
        ("PE", "Entropy"),
        ("P(True)", "PTrue_VLM"),
        ("Ecc. Unc.", "EccentricityUncertainty"),
        ("Img. Per.", "Image_Perturbation"),
        (r"$\text{LARS}_{Base}$", "LARS_BASE"),
    ]

    for display_name, metric_col in within_baseline_rows:
        row = {"label": display_name}

        for model in MODELS:
            for gen_ret in GEN_ORDER_GENERALIZATION:
                for dataset in DATASETS:
                    metric_v = get_within_metric(
                        results=results,
                        eval_dataset=dataset,
                        eval_model=model,
                        gen_retrieval=gen_ret,
                        metric_col=metric_col,
                    )
                    v = fmt3(metric_v) if pd.notna(metric_v) else "-"
                    row[(model, gen_ret, dataset)] = v

        # NEW: average across all 12 table entries for this row
        avg_vals = [
            extract_numeric_prefix(row.get(col, np.nan))
            for col in columns
        ]
        avg_vals = [v for v in avg_vals if pd.notna(v)]
        row[avg_col] = fmt3(np.mean(avg_vals)) if avg_vals else "-"

        single_rows.append(row)

    lars_row = {"label": r"$\text{LARS}_{Ret.}$"}
    mmra_row_top = {"label": r"\textbf{$\text{MMRA}$}"}
    mmra_row_bottom = {"label": ""}

    for model in MODELS:
        for gen_ret in GEN_ORDER_GENERALIZATION:
            retr = {
                "bm25": "bm25",
                "eva_clip+contriever": "eva_clip",
                "rerank": "rerank",
            }[gen_ret]

            for dataset in DATASETS:
                lars_col = train_col(model, dataset, retr, with_context=False)
                mmra_col = train_col(model, dataset, retr, with_context=True)

                lars_v = get_within_metric(
                    results=results,
                    eval_dataset=dataset,
                    eval_model=model,
                    gen_retrieval=gen_ret,
                    metric_col=lars_col,
                )
                mmra_v = get_within_metric(
                    results=results,
                    eval_dataset=dataset,
                    eval_model=model,
                    gen_retrieval=gen_ret,
                    metric_col=mmra_col,
                )

                lars_cell = fmt3(lars_v) if pd.notna(lars_v) else "-"
                mmra_top_cell = fmt3(mmra_v) if pd.notna(mmra_v) else "-"
                mmra_bottom_cell = format_mmra_gain_only(mmra_v, lars_v)

                sig_key = (model, dataset, retr, gen_ret)
                mmra_top_cell = append_sig_superscripts(
                    mmra_top_cell,
                    add_dag=dag_map.get(sig_key, False),
                    add_ddag=ddag_map.get(sig_key, False),
                )

                lars_row[(model, gen_ret, dataset)] = lars_cell
                mmra_row_top[(model, gen_ret, dataset)] = mmra_top_cell
                mmra_row_bottom[(model, gen_ret, dataset)] = mmra_bottom_cell

    # NEW: averages for LARS_Ret., MMRA, and MMRA gain row
    lars_avg_vals = [extract_numeric_prefix(lars_row.get(col, np.nan)) for col in columns]
    lars_avg_vals = [v for v in lars_avg_vals if pd.notna(v)]
    lars_row[avg_col] = fmt3(np.mean(lars_avg_vals)) if lars_avg_vals else "-"

    mmra_avg_vals = [extract_numeric_prefix(mmra_row_top.get(col, np.nan)) for col in columns]
    mmra_avg_vals = [v for v in mmra_avg_vals if pd.notna(v)]
    mmra_row_top[avg_col] = fmt3(np.mean(mmra_avg_vals)) if mmra_avg_vals else "-"

    mmra_gain_avg_vals = [extract_numeric_prefix(mmra_row_bottom.get(col, np.nan)) for col in columns]
    mmra_gain_avg_vals = [v for v in mmra_gain_avg_vals if pd.notna(v)]
    if mmra_gain_avg_vals:
        delta_avg = float(np.mean(mmra_gain_avg_vals))
        if delta_avg > 0:
            mmra_row_bottom[avg_col] = rf"\scriptsize \textcolor{{ForestGreen}}{{(+{delta_avg:.3f})}}"
        elif delta_avg < 0:
            mmra_row_bottom[avg_col] = rf"\scriptsize \textcolor{{BrickRed}}{{({delta_avg:.3f})}}"
        else:
            mmra_row_bottom[avg_col] = rf"\scriptsize \textcolor{{black}}{{(+0.000)}}"
    else:
        mmra_row_bottom[avg_col] = "-"

    single_rows.append(lars_row)

    # NEW: include Avg. in bolding
    rows_for_bolding = single_rows + [mmra_row_top]
    rows_for_bolding = bold_best_per_column(
        rows_for_bolding,
        columns=columns + [avg_col],
        exclude_labels={"Accuracy"}
    )

    n_single = len(single_rows)
    single_rows_bolded = rows_for_bolding[:n_single]
    mmra_row_top_bolded = rows_for_bolding[n_single]

    lines = []
    lines.append(r"% === Table 2: In-domain performance ==========")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3.2pt}")
    lines.append(
        r"\caption{Within-distribution AUROC performance across EVQA and InfoSeek for LLaVA1.5-7B and Qwen3-VL-4B. Only the BM25, EVAC., and BM25+MLM generation settings are shown. $\text{LARS}_{Ret.}$ uses the finetuned LARS model matched to the corresponding retrieval setting, and \textbf{$\text{MMRA}$} uses the corresponding MMRA-LARS model. The second MMRA row reports the gain or loss relative to $\text{LARS}_{Ret.}$. Superscripts \textsuperscript{\dag} and \textsuperscript{\ddag} denote statistically significant differences according to the DeLong test ($p<0.05$), compared to $\text{LARS}_{Base}$ and $\text{LARS}_{Ret.}$, respectively.}"
    )
    lines.append(r"\label{tab:within_distribution}")
    lines.append(r"\shrink")
    # NEW: one extra rightmost column
    lines.append(r"\begin{tabular}{l|cc|cc|cc|cc|cc|cc|c}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{LLM} & \multicolumn{6}{c|}{\textbf{LLaVA1.5-7B}} & \multicolumn{6}{c|}{\textbf{Qwen3-VL-4B}} & \textbf{Avg.} \\ \hline"
    )
    lines.append(
        r"\textbf{Retriever} & \multicolumn{2}{c}{\textbf{BM25}} & \multicolumn{2}{c}{\textbf{EVAC.}} & \multicolumn{2}{c|}{\textbf{BM25+MLM}} & \multicolumn{2}{c}{\textbf{BM25}} & \multicolumn{2}{c}{\textbf{EVAC.}} & \multicolumn{2}{c|}{\textbf{BM25+MLM}} & \\ \hline"
    )
    lines.append(
        r"\textbf{} & EVQA & InfoSeek & EVQA & InfoSeek & EVQA & InfoSeek & EVQA & InfoSeek & EVQA & InfoSeek & EVQA & InfoSeek & \\ \hline"
    )

    for row in single_rows_bolded:
        cells = [row["label"]]
        for col in columns:
            cells.append(render_value(row.get(col, "-")))
        cells.append(render_value(row.get(avg_col, "-")))
        lines.append(" & ".join(cells) + r" \\")

    cells = [mmra_row_top_bolded["label"]]
    for col in columns:
        cells.append(render_value(mmra_row_top_bolded.get(col, "-")))
    cells.append(render_value(mmra_row_top_bolded.get(avg_col, "-")))
    lines.append(" & ".join(cells) + r" \\")

    cells = [mmra_row_bottom["label"]]
    for col in columns:
        cells.append(render_value(mmra_row_bottom.get(col, "-")))
    cells.append(render_value(mmra_row_bottom.get(avg_col, "-")))
    lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\miniskip")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ============================================================
# TABLE 3: OOD RETRIEVER
# ============================================================

def build_ood_retriever_rows_for_model(results, model, dag_map, ddag_map):
    rows = []

    base_row = {"label": format_ood_method_label("lars", None)}
    for dataset in DATASETS:
        for gen_ret in GEN_ORDER:
            base_row[(dataset, gen_ret)] = fmt3(
                get_metric(results, dataset, model, gen_ret, "LARS_BASE")
            )
    rows.append(base_row)

    for retr in TRAIN_RETRIEVALS:
        matched_gen = gen_ret_from_train_retrieval(retr)

        lars_row = {"label": format_ood_method_label("lars", retr)}
        mmra_row = {"label": format_ood_method_label("mmra", retr)}

        for dataset in DATASETS:
            lars_col = train_col(model, dataset, retr, with_context=False)
            mmra_col = train_col(model, dataset, retr, with_context=True)

            for gen_ret in GEN_ORDER:
                lars_cell = fmt3(get_metric(results, dataset, model, gen_ret, lars_col))
                mmra_cell = fmt3(get_metric(results, dataset, model, gen_ret, mmra_col))

                sig_key = (model, dataset, retr, gen_ret)
                mmra_cell = append_sig_superscripts(
                    mmra_cell,
                    add_dag=dag_map.get(sig_key, False),
                    add_ddag=ddag_map.get(sig_key, False),
                )

                if gen_ret == matched_gen:
                    lars_cell = rf"\cellcolor{{gray!15}}{lars_cell}" if lars_cell != "-" else lars_cell
                    mmra_cell = rf"\cellcolor{{gray!15}}{mmra_cell}" if mmra_cell != "-" else mmra_cell

                lars_row[(dataset, gen_ret)] = lars_cell
                mmra_row[(dataset, gen_ret)] = mmra_cell

        rows.append(lars_row)
        rows.append(mmra_row)

    return rows


def build_ood_retriever_table_latex(results, dag_map, ddag_map):
    columns = get_table_columns()
    colspec = build_fixed_width_colspec(
        n_data_cols=12,
        label_width=TABLE3_LABEL_COL_WIDTH,
        data_width=TABLE3_DATA_COL_WIDTH,
    )

    llava_rows = build_ood_retriever_rows_for_model(results, "llava", dag_map, ddag_map)
    qwen_rows = build_ood_retriever_rows_for_model(results, "qwen", dag_map, ddag_map)

    lines = []
    lines.append(r"% === Table 3: OOD, Retriever =================")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(
        r"\caption{Generalizability of MMRA across different retrieval models. The UQ model is trained using one retrieval model and tested with another retrieval model. Superscripts \textsuperscript{\dag} and \textsuperscript{\ddag} denote statistically significant differences according to the DeLong test ($p<0.05$), compared to $\text{LARS}_{Base}$ and the corresponding LARS model in the same row block, respectively.}"
    )
    lines.append(r"\label{tab:ood_retriever}")
    lines.append(r"\shrink")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\hline")
    lines.append(r"\textbf{LLM} & \multicolumn{6}{c|}{\textbf{EVQA}} & \multicolumn{6}{c}{\textbf{InfoSeek}} \\ \hline")

    header = [r"\textbf{}"]
    for dataset in DATASETS:
        for g in GEN_ORDER:
            header.append(format_two_line_header(get_gen_header(dataset, g)))
    lines.append(" & ".join(header) + r" \\ \hline")

    lines.append(r"\rowcolor{gray!20}\multicolumn{13}{l}{\textit{LLaVA1.5-7B}} \\")
    for idx, row in enumerate(llava_rows):
        cells = [row["label"]]
        for col in columns:
            cells.append(render_value(row.get(col, "-")))
        lines.append(" & ".join(cells) + r" \\")
        if idx == 0:
            lines.append(r"\hdashline")
            lines.append(r"\hdashline")
        elif idx in {2, 4}:
            lines.append(r"\hdashline")

    lines.append(r"\rowcolor{gray!20}\multicolumn{13}{l}{\textit{Qwen3-VL-4B}} \\")
    for idx, row in enumerate(qwen_rows):
        cells = [row["label"]]
        for col in columns:
            cells.append(render_value(row.get(col, "-")))
        lines.append(" & ".join(cells) + r" \\")
        if idx == 0:
            lines.append(r"\hdashline")
            lines.append(r"\hdashline")
        elif idx in {2, 4}:
            lines.append(r"\hdashline")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ============================================================
# TABLE 4: OOD DATASET
# ============================================================

def build_ood_dataset_rows_for_model(results, model, ddag_map):
    lars_row = {"label": format_compact_setup_label("lars")}
    mmra_row = {"label": format_compact_setup_label("mmra")}

    transfer_pairs = [
        ("evqa", "infoseek"),
        ("infoseek", "evqa"),
    ]

    for src_dataset, tgt_dataset in transfer_pairs:
        for retr in TRAIN_RETRIEVALS:
            gen_ret = gen_ret_from_train_retrieval(retr)

            lars_v = get_metric(
                results, tgt_dataset, model, gen_ret, train_col(model, src_dataset, retr, with_context=False)
            )
            mmra_v = get_metric(
                results, tgt_dataset, model, gen_ret, train_col(model, src_dataset, retr, with_context=True)
            )

            lars_row[(src_dataset, tgt_dataset, retr)] = fmt3(lars_v)

            mmra_cell = fmt3(mmra_v)
            mmra_cell = append_sig_superscripts(
                mmra_cell,
                add_dag=False,
                add_ddag=ddag_map.get((model, src_dataset, tgt_dataset, retr), False),
            )
            mmra_row[(src_dataset, tgt_dataset, retr)] = mmra_cell

    return [lars_row, mmra_row]


def build_ood_dataset_table_latex(results, ddag_map):
    colspec = build_fixed_width_colspec(
        n_data_cols=6,
        label_width=TABLE45_LABEL_COL_WIDTH,
        data_width=TABLE45_DATA_COL_WIDTH,
    )

    lines = []
    lines.append(r"% === Table 4: OOD, Dataset ===================")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{2.4pt}")
    lines.append(
        r"\caption{Generalizability of MMRA across datasets. The UQ model is trained on one dataset and tested with another dataset. Superscript \textsuperscript{\ddag} denotes a statistically significant difference according to the DeLong test ($p<0.05$) compared to $\text{LARS}_{Ret.}$ in the same column.}"
    )
    lines.append(r"\label{tab:ood_dataset}")
    lines.append(r"\shrink")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Setup} & \multicolumn{3}{c|}{\textbf{EVQA $\rightarrow$ InfoSeek}} & \multicolumn{3}{c}{\textbf{InfoSeek $\rightarrow$ EVQA}} \\ \hline"
    )
    lines.append(
        r"\textbf{Ret.} & "
        + format_two_line_header("BM25") + " & "
        + format_two_line_header("EVAC.") + " & "
        + format_two_line_header("BM25+MLM") + " & "
        + format_two_line_header("BM25") + " & "
        + format_two_line_header("EVAC.") + " & "
        + format_two_line_header("BM25+MLM")
        + r" \\ \hline"
    )
    lines.append(r"\hline\hline")

    for model in MODELS:
        lines.append(
            rf"\rowcolor{{gray!20}}\multicolumn{{7}}{{c}}{{\textit{{{MODEL_LABELS[model]}}}}} \\"
        )
        rows = build_ood_dataset_rows_for_model(results, model, ddag_map)
        for row in rows:
            cells = [row["label"]]
            for key in [
                ("evqa", "infoseek", "bm25"),
                ("evqa", "infoseek", "eva_clip"),
                ("evqa", "infoseek", "rerank"),
                ("infoseek", "evqa", "bm25"),
                ("infoseek", "evqa", "eva_clip"),
                ("infoseek", "evqa", "rerank"),
            ]:
                cells.append(render_value(row.get(key, "-")))
            lines.append(" & ".join(cells) + r" \\")
        lines.append("")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ============================================================
# TABLE 5: OOD VLM
# ============================================================

def build_ood_vlm_rows_for_dataset(results, dataset, ddag_map):
    lars_row = {"label": format_compact_setup_label("lars")}
    mmra_row = {"label": format_compact_setup_label("mmra")}

    transfer_pairs = [
        ("llava", "qwen"),
        ("qwen", "llava"),
    ]

    for src_model, tgt_model in transfer_pairs:
        for retr in TRAIN_RETRIEVALS:
            gen_ret = gen_ret_from_train_retrieval(retr)

            lars_v = get_metric(
                results, dataset, tgt_model, gen_ret, train_col(src_model, dataset, retr, with_context=False)
            )
            mmra_v = get_metric(
                results, dataset, tgt_model, gen_ret, train_col(src_model, dataset, retr, with_context=True)
            )

            lars_row[(src_model, tgt_model, retr)] = fmt3(lars_v)

            mmra_cell = fmt3(mmra_v)
            mmra_cell = append_sig_superscripts(
                mmra_cell,
                add_dag=False,
                add_ddag=ddag_map.get((dataset, src_model, tgt_model, retr), False),
            )
            mmra_row[(src_model, tgt_model, retr)] = mmra_cell

    return [lars_row, mmra_row]


def build_ood_vlm_table_latex(results, ddag_map):
    colspec = build_fixed_width_colspec(
        n_data_cols=6,
        label_width=TABLE45_LABEL_COL_WIDTH,
        data_width=TABLE45_DATA_COL_WIDTH,
    )

    lines = []
    lines.append(r"% === Table 5: OOD, VLM =======================")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{2.4pt}")
    lines.append(
        r"\caption{Generalizability of MMRA across LLMs. The UQ model is trained on one LLM and tested with another LLM. Superscript \textsuperscript{\ddag} denotes a statistically significant difference according to the DeLong test ($p<0.05$) compared to $\text{LARS}_{Ret.}$ in the same column.}"
    )
    lines.append(r"\label{tab:ood_vlm}")
    lines.append(r"\shrink")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\hline")
    lines.append(
        r"\textbf{Setup} & \multicolumn{3}{c|}{\textbf{LLaVA $\rightarrow$ Qwen3}} & \multicolumn{3}{c}{\textbf{Qwen3 $\rightarrow$ LLaVA}} \\ \hline"
    )
    lines.append(
        r"\textbf{Ret.} & "
        + format_two_line_header("BM25") + " & "
        + format_two_line_header("EVAC.") + " & "
        + format_two_line_header("BM25+MLM") + " & "
        + format_two_line_header("BM25") + " & "
        + format_two_line_header("EVAC.") + " & "
        + format_two_line_header("BM25+MLM")
        + r" \\ \hline"
    )
    lines.append(r"\hline\hline")

    for dataset in DATASETS:
        lines.append(
            rf"\rowcolor{{gray!20}}\multicolumn{{7}}{{c}}{{\textit{{{DATASET_LABELS[dataset]}}}}} \\"
        )
        rows = build_ood_vlm_rows_for_dataset(results, dataset, ddag_map)
        for row in rows:
            cells = [row["label"]]
            for key in [
                ("llava", "qwen", "bm25"),
                ("llava", "qwen", "eva_clip"),
                ("llava", "qwen", "rerank"),
                ("qwen", "llava", "bm25"),
                ("qwen", "llava", "eva_clip"),
                ("qwen", "llava", "rerank"),
            ]:
                cells.append(render_value(row.get(key, "-")))
            lines.append(" & ".join(cells) + r" \\")
        lines.append("")

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\miniskip")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ============================================================
# BUILD ALL TABLES
# ============================================================

def build_all_tables(
    results,
    within_dag_map,
    within_ddag_map,
    ood_retr_dag_map,
    ood_retr_ddag_map,
    ood_dataset_ddag_map,
    ood_vlm_ddag_map,
):
    blocks = []

    blocks.append(r"\section{Results}")
    blocks.append(r"% Requires in preamble:")
    blocks.append(r"% \usepackage[table,dvipsnames]{xcolor}")
    blocks.append(r"% \usepackage{array}")
    blocks.append(r"% \usepackage{arydshln}")
    blocks.append("")
    blocks.append(r"% Also used by these tables:")
    blocks.append(r"% \newcommand{\shrink}{\resizebox{\columnwidth}{!}}")
    blocks.append(r"% For table* environments, your project may instead define \shrink with \textwidth.")
    blocks.append(r"% If \shrink is already defined in your project, keep your existing definition.")
    blocks.append("")

    blocks.append(build_big_table_latex(results))
    blocks.append("")
    blocks.append(build_within_distribution_table_latex(results, within_dag_map, within_ddag_map))
    blocks.append("")
    blocks.append(build_ood_retriever_table_latex(results, ood_retr_dag_map, ood_retr_ddag_map))
    blocks.append("")
    blocks.append(build_ood_dataset_table_latex(results, ood_dataset_ddag_map))
    blocks.append("")
    blocks.append(build_ood_vlm_table_latex(results, ood_vlm_ddag_map))
    blocks.append("")

    return "\n\n".join(blocks)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    results, raw_results = build_results_map(ROOT_DIR)

    expected = 4 * 6
    print(f"Found {len(results)}/{expected} experiment folders with readable results.")

    if not results:
        raise ValueError("No experiment results found. Check ROOT_DIR.")

    within_dag_map, within_ddag_map, delong_summary_df = compute_within_distribution_significance(raw_results)
    ood_retr_dag_map, ood_retr_ddag_map = compute_ood_retriever_significance(raw_results)
    ood_dataset_ddag_map = compute_ood_dataset_significance(raw_results)
    ood_vlm_ddag_map = compute_ood_vlm_significance(raw_results)

    tex = build_all_tables(
        results=results,
        within_dag_map=within_dag_map,
        within_ddag_map=within_ddag_map,
        ood_retr_dag_map=ood_retr_dag_map,
        ood_retr_ddag_map=ood_retr_ddag_map,
        ood_dataset_ddag_map=ood_dataset_ddag_map,
        ood_vlm_ddag_map=ood_vlm_ddag_map,
    )

    with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"Wrote LaTeX to: {os.path.abspath(OUTPUT_TEX)}")

    print("\n" + "=" * 100)
    print("WITHIN-DISTRIBUTION DeLong SUMMARY (matched LARS vs matched MMRA)")
    print("=" * 100)

    if delong_summary_df.empty:
        print("No valid within-distribution DeLong comparisons were found.")
    else:
        display_cols = [
            "eval_model_label",
            "eval_dataset_label",
            "gen_retrieval",
            "finetune_retrieval_label",
            "n",
            "n_pos",
            "n_neg",
            "auc_lars",
            "auc_mmra",
            "diff_mmra_minus_lars",
            "z_score",
            "p_value",
            "significant",
            "method",
            "var_diff",
        ]

        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.max_rows", None)

        print(delong_summary_df[display_cols].to_string(index=False))

        n_sig = int(delong_summary_df["significant"].sum())
        n_total = int(len(delong_summary_df))
        print("\n" + "-" * 100)
        print(f"Significant pairs (p < {DELONG_P_THRESHOLD}): {n_sig}/{n_total}")
