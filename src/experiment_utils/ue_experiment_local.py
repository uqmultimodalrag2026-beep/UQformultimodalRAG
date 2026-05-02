import os

# Set output dir to local
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
from TruthTorchLM.evaluators.substring_match import SubstringMatch
from TruthTorchLM.evaluators.exact_match import ExactMatch ##ToDo fix this, this is bugged
from TruthTorchLM.truth_methods import PTrue, EccentricityUncertainty
from TruthTorchLM import generate_with_truth_value, evaluate_truth_method
from huggingface_hub import login
import torch
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score





def load_model(model_name = "meta-llama/Meta-Llama-3-8B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_8bit=True,       # quantize to 8-bit have to use quantization
        device_map="cuda"        # can use auto too but cuda works with rtx4080 12GB vram
    )
    return model, tokenizer

def load_truth_methods():
    ptrue = PTrue()
    ecc_unc = EccentricityUncertainty()

    truth_methods = [#ptrue, 
                     ecc_unc]

    return truth_methods

def ask_question(question, model, tokenizer, truth_methods):
    chat = [
        {"role": "system", "content": "You are a helpful assistant. Give a single word answer."},
        {"role": "user", "content": question},
    ]

    output = generate_with_truth_value(
        model=model,
        tokenizer=tokenizer,
        messages=chat,
        truth_methods=truth_methods,
        max_new_tokens=100,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
    )
    return output

def load_dataset():
    df = pd.read_csv("data/popqa.tsv", sep="\t")
    return df

def run_experiment(model, truth_methods, tokenizer):
    #ToDo: Fix EM
    model_judge = SubstringMatch()
    results = evaluate_truth_method(dataset = 'pop_qa', model = model, truth_methods=truth_methods, 
    eval_metrics = ['auroc', 'prr'], tokenizer = tokenizer, size_of_data = 10, correctness_evaluator = model_judge, 
    return_method_details = True,  batch_generation = True, #wandb_push_method_details = False,
    max_new_tokens = 64, do_sample = True, seed = 0, pad_token_id=tokenizer.eos_token_id)
    correctness = results['output_dict']['generations_correctness']
    truth_values_p_true = results['output_dict']['truth_method_0']['truth_values']
    truth_values_ecc = results['output_dict']['truth_method_1']['truth_values']

    # Compute ROC curve points and AUC
    fpr_p, tpr_p, _ = roc_curve(correctness, truth_values_p_true)
    auc_p = roc_auc_score(correctness, truth_values_p_true)

    fpr_e, tpr_e, _ = roc_curve(correctness, truth_values_ecc)
    auc_e = roc_auc_score(correctness, truth_values_ecc)

    # Plot ROC curves
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_p, tpr_p, label=f"p_true (AUC = {auc_p:.3f})")
    plt.plot(fpr_e, tpr_e, label=f"ecc (AUC = {auc_e:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.5)")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    plot_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(plot_path)
    plt.close()
    for i in range(len(results['eval_list'])):
        print(results['output_dict']['truth_methods'][i],results['eval_list'][i])


model, tokenizer = load_model()
#for attr, value in vars(tokenizer).items():
#        print(f"{attr}: {value}")
truth_methods = load_truth_methods()
output =  ask_question("What is the capital of France?", model, tokenizer, truth_methods)
print(output)
#run_experiment(model, truth_methods, tokenizer)


