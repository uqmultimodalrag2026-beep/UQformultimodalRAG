
#!/usr/bin/env python3
"""
Save a dataset returned by `get_dataset(...)` as a pickle file.

Example usage:
    python save_dataset_as_pkl.py \
        --dataset trivia_qa \
        --split test \
        --size_of_data 0.1 \
        --seed 0 \
        --output trivia_qa_test.pkl

EVQA example:
    python save_dataset_as_pkl.py \
        --dataset evqa \
        --size_of_data 500 \
        --context_mode doc+ \
        --dataset_path /path/to/e_vqa \
        --image_directory /path/to/e_vqa \
        --dataset_csv evqa_small.csv \
        --retrieval_file bm25_docs_large_without_caption.csv \
        --output evqa_docplus.pkl

InfoSeek example:
    python save_dataset_as_pkl.py \
        --dataset infoseek \
        --size_of_data 500 \
        --context_mode bm25 \
        --dataset_path /path/to/infoseek \
        --image_directory /path/to/images \
        --dataset_csv infoseek_val_evidence.csv \
        --retrieval_file bm25_docs_large_without_caption.csv \
        --kb_path /path/to/e_vqa \
        --output infoseek_bm25.pkl
"""
from TruthTorchLM.utils.dataset_utils import get_dataset
import argparse
import os
import pickle
from typing import Any



def parse_size_of_data(value: str) -> Any:
    """
    Parse size_of_data from CLI.

    Supports:
    - float fractions like "0.1" or "1.0"
    - integer counts like "500"
    - Python-style range notation like "range(0,100)" or "0:100"
    """
    value = value.strip()

    if value.startswith("range(") and value.endswith(")"):
        inner = value[len("range("):-1]
        parts = [p.strip() for p in inner.split(",")]
        if len(parts) == 1:
            return range(int(parts[0]))
        if len(parts) == 2:
            return range(int(parts[0]), int(parts[1]))
        if len(parts) == 3:
            return range(int(parts[0]), int(parts[1]), int(parts[2]))
        raise ValueError(f"Invalid range format: {value}")

    if ":" in value:
        parts = [p.strip() for p in value.split(":")]
        if len(parts) == 2:
            return range(int(parts[0]), int(parts[1]))
        if len(parts) == 3:
            return range(int(parts[0]), int(parts[1]), int(parts[2]))
        raise ValueError(f"Invalid colon range format: {value}")

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError as e:
        raise ValueError(
            f"Could not parse size_of_data='{value}'. "
            f"Use float (e.g. 0.1), int (e.g. 500), or range (e.g. 0:100)."
        ) from e


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a dataset with get_dataset(...) and save it as a .pkl file."
    )

    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. trivia_qa, gsm8k, evqa, infoseek")
    parser.add_argument("--output", required=True, help="Output .pkl file path")
    parser.add_argument("--size_of_data", default="1.0", help="Fraction, count, or range. Examples: 1.0, 500, 0:100")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--split", default="test", help="Dataset split, e.g. test or train")
    parser.add_argument("--context_mode", default=None, help="Context mode for EVQA/InfoSeek")
    parser.add_argument("--dataset_path", default=None, help="Path to dataset directory")
    parser.add_argument("--image_directory", default=None, help="Path to image directory")
    parser.add_argument("--dataset_csv", default=None, help="CSV filename to use")
    parser.add_argument("--retrieval_file", default=None, help="Retrieval CSV filename")
    parser.add_argument("--kb_path", default=None, help="Knowledge base path for InfoSeek")
    parser.add_argument(
        "--protocol",
        type=int,
        default=pickle.HIGHEST_PROTOCOL,
        help="Pickle protocol version",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    size_of_data = parse_size_of_data(args.size_of_data)

    print("Calling get_dataset(...)")
    dataset = get_dataset(
        dataset=args.dataset,
        size_of_data=size_of_data,
        seed=args.seed,
        split=args.split,
        context_mode=args.context_mode,
        dataset_path=args.dataset_path,
        image_directory=args.image_directory,
        dataset_csv=args.dataset_csv,
        retrieval_file=args.retrieval_file,
        kb_path=args.kb_path,
    )

    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "wb") as f:
        pickle.dump(dataset, f, protocol=args.protocol)

    print(f"Saved dataset to: {args.output}")
    print(f"Number of samples: {len(dataset)}")


if __name__ == "__main__":
    main()
