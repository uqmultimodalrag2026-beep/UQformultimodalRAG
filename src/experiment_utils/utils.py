import os
from transformers import AutoProcessor, LlavaForConditionalGeneration,Qwen3VLForConditionalGeneration, BitsAndBytesConfig
from TruthTorchLM.evaluators.substring_match import SubstringMatch
from TruthTorchLM.evaluators.exact_match import ExactMatch 
from TruthTorchLM.evaluators.model_judge import ModelJudge 
from TruthTorchLM.truth_methods import EccentricityUncertainty
from TruthTorchLM.truth_methods.p_true_vlm import PTrue_VLM
from TruthTorchLM.truth_methods.bm25_score import BM25_score
from TruthTorchLM.truth_methods.image_pertubation_retrieval import Image_pertubation_retrieval_score
from TruthTorchLM.truth_methods.entropy import Entropy
from TruthTorchLM.truth_methods.semantic_entropy import SemanticEntropy
from TruthTorchLM.truth_methods.image_perturbation import ImagePerturbation
from TruthTorchLM.truth_methods.p_relevant_vlm import PRelevant_VLM
from TruthTorchLM.truth_methods.oracle_relevance import Oracle_relevance
from TruthTorchLM.truth_methods.lars_vlm_base import LARS_BASE
from TruthTorchLM.truth_methods.lars_vlm_finetuned import LARS_Finetuned
from TruthTorchLM.truth_methods.lemuq import LeMUQ
from TruthTorchLM.truth_methods.lemuq_ablated import LeMUQ_ablated
from TruthTorchLM.generation_vlm import generate_with_truth_value
from TruthTorchLM.evaluators.eval_truth_method_vlm import evaluate_truth_method
from TruthTorchLM.templates import DEFAULT_LLAVA_PROMPT_NO_CONTEXT, DEFAULT_LLAVA_PROMPT_WITH_CONTEXT, DEFAULT_QWEN_PROMPT_NO_CONTEXT, DEFAULT_QWEN_PROMPT_WITH_CONTEXT
from huggingface_hub import login
import glob
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

def load_model(model_name="llava-hf/llava-1.5-7b-hf", load_in_8bit=True):
    if model_name == "llava-hf/llava-1.5-7b-hf":
        # BitsAndBytes quantization config for 8-bit
        quant_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)

        # Load processor
        processor = AutoProcessor.from_pretrained(model_name)

        # Load model in 8-bit
        model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto"
        )
        return model, processor
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",           # automatically chooses FP16/FP32 depending on device
        device_map="auto"       # automatically places layers on GPU/CPU
    )
        processor = AutoProcessor.from_pretrained(model_name)
        return model, processor


def load_truth_methods(truth_method_names, generations = 10, lars_dicts = []):
    truth_methods = []
    if "eccunc" in truth_method_names:
        truth_methods.append(EccentricityUncertainty(number_of_generations=generations))
    if "ptrue" in truth_method_names:
        truth_methods.append(PTrue_VLM(number_of_ideas = generations, with_context=False))
    if "bm25_score" in truth_method_names:
        truth_methods.append(BM25_score())
    if "image_pertubation_retrieval_score" in truth_method_names:
        truth_methods.append(Image_pertubation_retrieval_score())
    if "entropy" in truth_method_names:
        truth_methods.append(Entropy(number_of_generations=generations))
    if "semantic_entropy" in truth_method_names:
        truth_methods.append(SemanticEntropy(number_of_generations=generations))
    if "image_pertubation" in truth_method_names:
        truth_methods.append(ImagePerturbation(number_of_ideas = generations, with_context=False))
    if "prelevant" in truth_method_names:
        truth_methods.append(PRelevant_VLM())
    if "oracle_relevance" in truth_method_names:
        truth_methods.append(Oracle_relevance())
    if "lars" in truth_method_names:
        truth_methods.append(LARS_BASE())
    for dict in lars_dicts:
        lars_name = dict['lars_name']
        version = dict['lars_version']
        lars_model_path = dict['model_path']
        lars_with_context = dict['lars_with_context']
        if "lars_base" == version:
            truth_methods.append(LARS_BASE(lars_with_context=lars_with_context))
        if "lemuq_ablated" == version:
            truth_methods.append(LeMUQ_ablated(model_path=lars_model_path,lars_with_context=lars_with_context, name = lars_name, p_prime=dict['p_prime'], p_q_i=dict['p_q_i'], p_q_c=dict['p_q_c'], p_q=dict['p_q']))
        if "lemuq" == version:
            truth_methods.append(LeMUQ(model_path=lars_model_path,lars_with_context=lars_with_context, name = lars_name))
        if "lars_finetuned" == version:
            truth_methods.append(LARS_Finetuned(model_path=lars_model_path,lars_with_context=lars_with_context, name = lars_name))

    if len(truth_methods) != len(truth_method_names) + len(lars_dicts):
        raise ValueError(f"There was an error in the truth methods. {truth_method_names}, {len(truth_methods), {len(truth_method_names)}}")
    return truth_methods

def load_model_judge(model_judge_name = None):        
    if model_judge_name == "meta-llama/Meta-Llama-3-8B-Instruct":
        model_for_eval = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype=torch.bfloat16).to('cuda:0')
        tokenizer_for_eval = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_fast=False)
        return ModelJudge(model_for_eval, tokenizer_for_eval)
    print("Using EM as default model judge.")
    return ExactMatch()

def run_experiment(model, processor, prompt_template, truth_methods, dataset, context_mode, samples, output_dir, model_judge, dataset_path, image_directory, dataset_csv, retrieval_csv, kb_path):    
    results = evaluate_truth_method(dataset = dataset, context_mode = context_mode, model = model, truth_methods=truth_methods, 
    eval_metrics = ['auroc', 'prr'], processor = processor, size_of_data = samples, correctness_evaluator = model_judge, 
    return_method_details = True,  batch_generation = False,
    max_new_tokens = 64, do_sample = True, seed = 0, eos_token_id = processor.tokenizer.eos_token_id, 
    prompt_template = prompt_template, dataset_path = dataset_path, image_directory = image_directory,
      dataset_csv=dataset_csv, retrieval_csv=retrieval_csv, kb_path= kb_path)
    correctness = results['output_dict']['generations_correctness']

    truth_values = []
    for i in range(len(truth_methods)):
        truth_values.append(results['output_dict'][f"truth_method_{i}"]['truth_values'])

    accuracy = np.mean(correctness)
    print(f"Accuracy: {accuracy:.3f}")
    for i in range(len(results['eval_list'])):
        print(results['output_dict']['truth_methods'][i],results['eval_list'][i])

    rows = []
    if isinstance(samples, range):
        n_rows = len(samples)
    else:
        n_rows = samples
    
    for i in range(n_rows):
        question_text = results['output_dict']['question_text'][i]
        prediction = results['output_dict']['generations'][i]
        ground_truths = results['output_dict']['ground_truths'][i]
        #
        if results['output_dict']['truth_method_0']['name'].split()[0] == "EccentricityUncertainty":
            generated_text = results['output_dict']['truth_method_0']['method_specific_details'][i]['generated_texts']
        else:
            generated_text = []
        is_correct = correctness[i]
        entry = {
            "question_text": question_text,
            "prediction": prediction,
            "ground_truths": ground_truths,
            "generated_text": generated_text,
            "correctness": is_correct
        }
        if context_mode:
            context = results['output_dict']['contexts'][i]
            entry['context'] = str(context).replace('\n', ' ').replace('\r', ' ') #remove linebreaks for formatting
        rows.append(entry)

    df = pd.DataFrame(rows)
    for i in range(len(truth_methods)):
        method_name = results['output_dict'][f'truth_method_{i}']['name'].split()[0]
        df[method_name] = truth_values[i]
    
    os.makedirs(output_dir, exist_ok=True)

    # Define unified results path
    csv_path = os.path.join(output_dir, "results.csv")

    # If results.csv exists, append; otherwise create a new one
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(csv_path, index=False)
        print(f"Appended {len(df)} rows to existing results file: {csv_path}")
    else:
        df.to_csv(csv_path, index=False)
        print(f"Created new results file: {csv_path}")


def run_experiment_with_intermediate_saves(
    model, processor, prompt_template, truth_methods,
    dataset, context_mode, samples, output_dir, samples_per_run, model_judge, dataset_path, image_directory, dataset_csv, retrieval_csv, kb_path
):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    num_runs = math.ceil(samples / samples_per_run)

    for run_idx in range(num_runs):
        start_sample = run_idx * samples_per_run
        end_sample = min(start_sample + samples_per_run, samples)

        run_samples = range(start_sample, end_sample)

        print(f"Running experiment for samples {start_sample} to {end_sample - 1}")

        run_experiment(
            model=model,
            processor=processor,
            prompt_template=prompt_template,
            truth_methods=truth_methods,
            dataset=dataset,
            context_mode = context_mode,
            samples=run_samples,  # always pass a range here
            output_dir=output_dir,
            model_judge=model_judge,
            dataset_path = dataset_path,
            image_directory = image_directory,
            dataset_csv=dataset_csv,
            retrieval_csv=retrieval_csv,
            kb_path=kb_path
        )

def plot_results(results_folder):
    # Find all CSV files
    csv_files = glob.glob(os.path.join(results_folder, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {results_folder}")
        return

    # Append all CSVs into one DataFrame
    all_dfs = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        all_dfs.append(df)
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Extract correctness
    correctness = combined_df["correctness"].astype(int).values

    # Compute accuracy
    accuracy = np.mean(correctness)

    # Create figure
    plt.figure(figsize=(7, 6))

    # Loop through truth method columns (everything after correctness)
    for col in combined_df.columns:
        if col in ["question_text", "prediction", "ground_truths", "generated_text", "correctness", "context"]:
            continue

        truth_vals = combined_df[col].astype(float).values

        try:
            fpr, tpr, _ = roc_curve(correctness, truth_vals)
            auc = roc_auc_score(correctness, truth_vals)
            plt.plot(fpr, tpr, label=f"{col} (AUC = {auc:.3f})")
        except ValueError:
            print(f"Skipping {col} (not enough variation).")
            continue

    # Baseline random curve
    plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.5)")

    # Labels and title
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (Accuracy = {accuracy:.3f})")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    # Save single combined plot
    plot_path = os.path.join(results_folder, "combined_roc_curves.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved combined ROC plot → {plot_path}")

def run(config):
    model = config.model
    context_mode = config.context_mode
    if "llava" in model:
        if context_mode is None or context_mode == "None":
            prompt_template = DEFAULT_LLAVA_PROMPT_NO_CONTEXT
        else:
            prompt_template = DEFAULT_LLAVA_PROMPT_WITH_CONTEXT
    elif "Qwen" in model: ##asume QWEN-VL
        if context_mode is None or context_mode == "None":
            prompt_template = DEFAULT_QWEN_PROMPT_NO_CONTEXT
        else:
            prompt_template = DEFAULT_QWEN_PROMPT_WITH_CONTEXT
    model, processor = load_model(model, config.eight_bit)
    truth_methods = load_truth_methods(config.truth_methods, config.generations_per_sample, lars_dicts=config.lars_dicts)
    dataset = config.dataset
    
    samples = config.samples
    output_dir = config.output_dir
    samples_per_run = config.samples_per_run
    model_judge = load_model_judge(config.model_judge)
    dataset_path = config.dataset_path
    image_directory = config.image_directory
    dataset_csv = config.dataset_csv
    retrieval_csv = config.retrieval_csv
    kb_path = config.kb_path
    

    run_experiment_with_intermediate_saves(model, processor, prompt_template, truth_methods,
                                            dataset,context_mode, samples, output_dir, samples_per_run=samples_per_run, model_judge=model_judge,
                                              dataset_path=dataset_path, image_directory=image_directory, dataset_csv=dataset_csv, retrieval_csv=retrieval_csv, kb_path=kb_path)
    plot_results(output_dir)
