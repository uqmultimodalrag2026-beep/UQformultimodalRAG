#!/usr/bin/env python3

import argparse

from TruthTorchLM.evaluators.exact_match import ExactMatch
from TruthTorchLM.templates import (
    DEFAULT_LLAVA_PROMPT_NO_CONTEXT,
    DEFAULT_LLAVA_PROMPT_WITH_CONTEXT,
    DEFAULT_QWEN_PROMPT_NO_CONTEXT,
    DEFAULT_QWEN_PROMPT_WITH_CONTEXT
)
from TruthTorchLM.truth_methods.lars_vlm_base import LARS_BASE
from TruthTorchLM.truth_methods.lemuq import LeMUQ

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and save LeMUQ VLM training/validation data"
    )
    parser.add_argument(
        "--lars-version",
        type=str,
        default="base",
        help="Version of the truth method (base or lemuq)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="List of dataset names (e.g. evqa)",
    )
    parser.add_argument(
        "--size-for-each-dataset",
        nargs="+",
        type=int,
        required=True,
        help="Number of samples per dataset (must match datasets length)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--chat-model-name",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
    )
    parser.add_argument(
        "--context-mode",
        type=str,
        default=None,
        help="Context mode (e.g. doc+, None for no context)",
    )
    parser.add_argument(
        "--dataset-csv",
        type=str,
        default=None,
        help="Optional CSV path for datasets",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Root path to datasets",
    )
    parser.add_argument(
        "--kb-path",
        type=str,
        default=None,
        help="Root path to kb",
    )
    parser.add_argument(
        "--image-directory",
        type=str,
        default=None,
        help="Path to image directory",
    )

    parser.add_argument(
        "--retrieval-file",
        type=str,
        default=None,
        help="Retrieval file name (including .csv)",
    )
    parser.add_argument(
        "--save-data-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--load-data-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num-gen-per-question",
        type=int,
        default=5,
        help="Number of generations per question",
    )
    parser.add_argument(
        "--lars-with-context",
        action="store_true",
        help="Enable LeMUQ with context"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # === Context logic (EXACTLY as requested) ===
    context_mode = args.context_mode
    model = args.chat_model_name
    if "Qwen" in model:
        if context_mode is None:
            user_prompt = DEFAULT_QWEN_PROMPT_NO_CONTEXT
        else:
            user_prompt = DEFAULT_QWEN_PROMPT_WITH_CONTEXT
    else:
        if context_mode is None:
            user_prompt = DEFAULT_LLAVA_PROMPT_NO_CONTEXT
        else:
            user_prompt = DEFAULT_LLAVA_PROMPT_WITH_CONTEXT
    
        

    # === Initialize truth method and evaluator ===
    if args.lars_version == "base":            
        lars = LARS_BASE(lars_with_context = args.lars_with_context)
    elif args.lars_version == "lemuq":      
        lars = LeMUQ(lars_with_context = args.lars_with_context)
    else:
        raise ValueError("Unknown truth method version")
    correctness_evaluator = ExactMatch()
    # === Generate and save data ===
    lars.train_lars_model(
        datasets=args.datasets,
        size_for_each_dataset=args.size_for_each_dataset,
        val_ratio=args.val_ratio,
        seed=args.seed,
        chat_model_name=args.chat_model_name,
        correctness_evaluator=correctness_evaluator,
        context_mode=context_mode,
        user_prompt=user_prompt,
        dataset_csv=args.dataset_csv,
        dataset_path=args.dataset_path,
        kb_path = args.kb_path,
        image_directory=args.image_directory,
        retrieval_file=args.retrieval_file,
        num_gen_per_question=args.num_gen_per_question,
        save_data_path=args.save_data_path,
        load_data_path=args.load_data_path,
        save_path=args.save_model_path,
        
    ) 


if __name__ == "__main__":
    main()
