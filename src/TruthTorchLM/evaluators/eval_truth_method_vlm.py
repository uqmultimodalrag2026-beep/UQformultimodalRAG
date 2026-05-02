from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Union
from TruthTorchLM.truth_methods import TruthMethod
from .correctness_evaluator import CorrectnessEvaluator
from .exact_match import ExactMatch
from TruthTorchLM.availability import AVAILABLE_EVALUATION_METRICS
from TruthTorchLM.templates import DEFAULT_SYSTEM_BENCHMARK_PROMPT, DEFAULT_USER_PROMPT
from TruthTorchLM.utils.dataset_utils import get_dataset
from TruthTorchLM.utils.eval_utils import metric_score, run_over_visual_dataset
import wandb


def evaluate_truth_method(
    dataset: Union[str, list],
    context_mode,
    model: Union[str, PreTrainedModel],
    truth_methods: list[TruthMethod],
    processor = None,
    eval_metrics: list[str] = ["auroc"],
    correctness_evaluator: CorrectnessEvaluator = ExactMatch(),
    size_of_data=1.0,
    previous_context = None, 
    user_prompt = None,
    prompt_template = None,
    seed: int = 0,
    return_method_details: bool = False,
    batch_generation=True,
    add_generation_prompt=False,
    continue_final_message=False,
    split="test",
    dataset_csv = None,
    retrieval_csv = None,
    kb_path = None,
    **kwargs,
):

    dataset_path = kwargs.pop("dataset_path")
    image_directory = kwargs.pop("image_directory")
    dataset = get_dataset(
        dataset, size_of_data=size_of_data, seed=seed, split=split, context_mode=context_mode, dataset_path=dataset_path, image_directory = image_directory, dataset_csv=dataset_csv, retrieval_file=retrieval_csv, kb_path=kb_path)
    kwargs["context_mode"] = context_mode


    for eval_metric in eval_metrics:
        if eval_metric not in AVAILABLE_EVALUATION_METRICS:
            raise ValueError(
                f"Evaluation metric {eval_metric} is not available. Available evaluation metrics are: {AVAILABLE_EVALUATION_METRICS}"
            )

    output_dict = run_over_visual_dataset(
        dataset,
        model,
        truth_methods,
        processor=processor,
        correctness_evaluator=correctness_evaluator,
        previous_context=previous_context,
        user_prompt=user_prompt,
        seed=seed,
        return_method_details=return_method_details,
        batch_generation=batch_generation,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
        prompt_template = prompt_template,
        **kwargs,
    )

    eval_list = get_metric_scores(
        output_dict=output_dict, eval_metrics=eval_metrics, seed=seed
    )

    return {"eval_list": eval_list, "output_dict": output_dict}


def get_metric_scores(output_dict: dict, eval_metrics: list[str], seed: int = 0):
    truth_methods = output_dict["truth_methods"]
    eval_list = []
    for i in range(len(truth_methods)):
        eval_dict = metric_score(
            eval_metrics,
            output_dict["generations_correctness"],
            output_dict[f"truth_method_{i}"]["truth_values"],
            output_dict[f"truth_method_{i}"]["normalized_truth_values"],
            seed=seed,
        )
        eval_list.append(eval_dict)
    return eval_list
