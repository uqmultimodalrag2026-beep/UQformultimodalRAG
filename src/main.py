from experiment_utils.utils import run, plot_results
from dataclasses import dataclass, field
from typing import List, Optional
import argparse
import json

@dataclass
class ExperimentConfig:
    # Core experiment settings
    model: str = "llava-hf/llava-1.5-7b-hf"
    truth_methods: List[str] = None
    dataset: str = "evqa"
    context_mode: str = None
    samples: int = 100
    output_dir: str = "./experiment_results/test"
    samples_per_run: int = 10
    generations_per_sample: int = 10
    # Optional model judge
    model_judge: Optional[str] = None
    # 8-bit option
    eight_bit: bool = True
    # Optional dataset-related paths
    dataset_path: Optional[str] = None
    image_directory: Optional[str] = None
    dataset_csv: Optional[str] = None
    retrieval_csv: Optional[str] = None
    kb_path: Optional[str] = None
    # LeMUQ configuration
    lars_dicts: List = field(default_factory=list)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="Run experiment")

    parser.add_argument("--model", type=str, default=ExperimentConfig.model)
    parser.add_argument(
        "--truth_methods",
        type=str,
        nargs="+",
        default=[],
        help="List of truth methods (space-separated)",
    )
    parser.add_argument("--dataset", type=str, default=ExperimentConfig.dataset)
    parser.add_argument("--context_mode", type=str, default=ExperimentConfig.context_mode)
    parser.add_argument("--samples", type=int, default=ExperimentConfig.samples)
    parser.add_argument("--output_dir", type=str, default=ExperimentConfig.output_dir)
    parser.add_argument("--samples_per_run", type=int, default=ExperimentConfig.samples_per_run)
    parser.add_argument(
        "--generations_per_sample",
        type=int,
        default=ExperimentConfig.generations_per_sample,
    )

    parser.add_argument(
        "--model_judge",
        type=str,
        default=None,
        help="Optional judge model, use meta-llama/Meta-Llama-3-8B-Instruct for llama",
    )

    parser.add_argument(
        "--no_8bit",
        action="store_false",
        dest="eight_bit",
        help="Disable 8-bit loading",
    )

    # Optional paths
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--image_directory",
        type=str,
        default=None,
        help="Path to image directory",
    )
    parser.add_argument(
        "--dataset_csv",
        type=str,
        default=None,
        help="Dataset CSV file name",
    )
    parser.add_argument(
        "--retrieval_csv",
        type=str,
        default=None,
        help="Retrieval CSV file name",
    )
    
    parser.add_argument(
        "--lars_dicts",
        type=str,
        default="[]",
        help="JSON list of LeMUQ dicts, e.g. "
            '[{"lars_name": "my_lemuq", "lars_version": "lemuq", "model_path": "/path", "lars_with_context": true}]',
    )
    parser.add_argument(
        "--kb_path",
        type=str,
        default=None,
        help="Path to kb root directory",
    )

    args = parser.parse_args()
    context_mode = None
    if args.context_mode != "None":
        context_mode = args.context_mode
    # Convert JSON string to Python list of dicts
    lars_dicts = json.loads(args.lars_dicts) if args.lars_dicts else []
    return ExperimentConfig(
        model=args.model,
        truth_methods=args.truth_methods,
        dataset=args.dataset,
        context_mode=context_mode,
        samples=args.samples,
        output_dir=args.output_dir,
        samples_per_run=args.samples_per_run,
        generations_per_sample=args.generations_per_sample,
        model_judge=args.model_judge,
        eight_bit=args.eight_bit,
        dataset_path=args.dataset_path,
        image_directory=args.image_directory,
        dataset_csv = args.dataset_csv,
        retrieval_csv= args.retrieval_csv,
        kb_path= args.kb_path,
        lars_dicts=lars_dicts
    )


if __name__ == "__main__":
    config = parse_args()
    """
    lars_dicts = [{
        "lars_version": "lemuq_ablated",
        "lars_name": "lemuq_ablated_test",
        "model_path": None,
        "lars_with_context": False,
        "p_prime": True,
        "p_q_i": True,
        "p_q_c": False,
        "p_q": True,
    }
    ]
    config = ExperimentConfig(samples=1,context_mode="doc+", samples_per_run=1,  truth_methods=["eccunc", "lars"] , model="Qwen/Qwen3-VL-2B-Instruct", 
                              lars_dicts=lars_dicts, output_dir = "./experiment_results/test") #
    #config = ExperimentConfig(dataset="infoseek", samples=1,context_mode="doc+", samples_per_run=1,  truth_methods=["eccunc", "lars"] )
    """
    run(config)
