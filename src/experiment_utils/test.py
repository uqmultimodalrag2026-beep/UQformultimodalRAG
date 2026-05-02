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

    truth_methods = [ptrue, ecc_unc]

    return truth_methods

def ask_question(question, model, tokenizer, truth_methods):
    chat = [
        {"role": "system", "content": "You are a helpful assistant. Give short and precise answers."},
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

model, tokenizer = load_model()
truth_methods = load_truth_methods()
ask_question("What is the capital of France?", model, tokenizer, truth_methods)