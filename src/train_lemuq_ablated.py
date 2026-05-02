#!/usr/bin/env python3

import argparse

from TruthTorchLM.evaluators.exact_match import ExactMatch
from TruthTorchLM.templates import (
    DEFAULT_LLAVA_PROMPT_NO_CONTEXT,
    DEFAULT_LLAVA_PROMPT_WITH_CONTEXT,
    DEFAULT_QWEN_PROMPT_NO_CONTEXT,
    DEFAULT_QWEN_PROMPT_WITH_CONTEXT
)

from TruthTorchLM.truth_methods.lemuq_ablated import LeMUQ_ablated

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and save LARS VLM training/validation data"
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
        "--p_prime",
        action="store_true",
        help="Enable p_prime"
    )

    parser.add_argument(
        "--p_q_i",
        action="store_true",
        help="Enable p_q_i"
    )

    parser.add_argument(
        "--p_q_c",
        action="store_true",
        help="Enable p_q_c"
    )

    parser.add_argument(
        "--p_q",
        action="store_true",
        help="Enable p_q"
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
    
        


    lars = LeMUQ_ablated(lars_with_context = True, p_prime=args.p_prime, p_q_i= args.p_q_i, p_q_c=args.p_q_c, p_q=args.p_q)

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
