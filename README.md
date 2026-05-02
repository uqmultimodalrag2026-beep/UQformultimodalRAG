# LeMUQ / TruthTorchLM-VLM

This repository is a forked copy of [TruthTorchLM](https://github.com/Ybakman/TruthTorchLM).

This fork contains the code for the paper:

**Uncertainty Quantification for Multimodal Retrieval Augmented Generation**

## Main Changes

This fork extends TruthTorchLM for multimodal retrieval augmented generation and vision-language models.

The main modifications are:

- **Generation support for VLMs**, including [Qwen3-VL](https://huggingface.co/collections/Qwen/qwen3-vl) and [LLaVA 1.5](https://huggingface.co/llava-hf/llava-1.5-7b-hf).
- **New and adapted VLM truth methods**:
  - [LARS VLM base](src/TruthTorchLM/truth_methods/lars_vlm_base.py)
  - [LARS VLM finetuned](src/TruthTorchLM/truth_methods/lars_vlm_finetuned.py)
  - [LeMUQ](src/TruthTorchLM/truth_methods/lemuq.py)
  - [LeMUQ ablated](src/TruthTorchLM/truth_methods/lemuq_ablated.py)
  - [P(Relevant) VLM](src/TruthTorchLM/truth_methods/p_relevant_vlm.py)
- **New dataset support** for [EVQA](https://github.com/google-research/google-research/tree/master/encyclopedic_vqa) and [InfoSeek](https://github.com/open-vision-language/infoseek) in [dataset_utils.py](src/TruthTorchLM/utils/dataset_utils.py).
- **Experiment pipeline scripts** for the paper in [main.py](src/main.py) and [experiment_utils](src/experiment_utils).
- **Analysis and plotting scripts** for the paper in [src/analysis](src/analysis), including [make_main_tables.py](src/analysis/make_main_tables.py) and [make_ablation_table.py](src/analysis/make_ablation_table.py).

## Usage

### Running Inference

[main.py](src/main.py) can be used to run inference and compare truth methods on a dataset.

The script supports loading VLMs, datasets, context/retrieval modes, and configured truth methods through command-line arguments.

### Training LeMUQ and VLM LARS Models

[train_lars.py](src/train_lars.py) can be used to finetune LARS adapted for VLM models and to train LeMUQ models.

[train_lemuq_ablated.py](src/train_lemuq_ablated.py) can be used to train the ablated LeMUQ models.

Example SLURM commands for recomputing the experiment results are provided as text files in this repository.

### VLM Evaluation

To run separate evaluations for VLM truth methods, use [eval_truth_method_vlm.py](src/TruthTorchLM/evaluators/eval_truth_method_vlm.py).

## Image Perturbation Features

The image perturbation scores used in the experiments were computed separately with [VisualUE](https://github.com/ErumMushtaq/VisualUE) and appended to the resulting experiment outputs.

## Notes on Data and Paths

The SLURM scripts use placeholder variables such as:

```bash
DATA_ROOT="/dataset_root_dir"
MODEL_ROOT="/model_root_dir"
RESULTS_ROOT="/result_root_dir/experiment_results_main"
LOG_DIR="/log_root_dir"
```

Edit these variables at the top of each script for your cluster. Dataset CSV and retrieval filenames are also defined through variables at the top of the scripts.

## License

This fork follows the original TruthTorchLM license. The license file has been anonymized for review. See [LICENSE](LICENSE).

## Citation

If you use this repository, please cite the paper:

```bibtex
@inproceedings{anonymous2026lemuq,
  title = {Uncertainty Quantification for Multimodal Retrieval Augmented Generation},
  author = {Anonymous Authors},
  booktitle = {To appear},
  year = {2026},
  url = {https://anonymous.4open.science/}
}
```

Please also cite the original TruthTorchLM repository where appropriate:

```bibtex
@misc{bakman2024truthtorchlm,
  title = {TruthTorchLM: A Comprehensive Package for Assessing/Predicting Truthfulness in LLM Outputs},
  author = {Bakman, Yavuz Faruk and others},
  year = {2024},
  url = {https://github.com/Ybakman/TruthTorchLM}
}
```
