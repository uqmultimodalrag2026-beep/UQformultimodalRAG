import os

# Set output dir to local
output_dir = "./plots"
os.makedirs(output_dir, exist_ok=True)
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from TruthTorchLM.evaluators.substring_match import SubstringMatch
from TruthTorchLM.evaluators.exact_match import ExactMatch 
from TruthTorchLM.truth_methods import EccentricityUncertainty
from TruthTorchLM.truth_methods.p_true_vlm import PTrue_VLM
from TruthTorchLM.generation_vlm import generate_with_truth_value
from huggingface_hub import login
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)




def load_model(model_name="llava-hf/llava-1.5-7b-hf"):
    # BitsAndBytes quantization config for 8-bit
    quant_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load processor
    processor = AutoProcessor.from_pretrained(model_name)

    # Load model in 8-bit
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto"
    )

    return model, processor

def load_truth_methods():
    ptrue = PTrue_VLM()
    ecc_unc = EccentricityUncertainty()

    truth_methods = [ecc_unc, ptrue]

    return truth_methods

def ask_question(question, image_path, model, processor, truth_methods):
    image = Image.open(image_path).convert("RGB")

    # Prepare prompt (LLaVA expects <image> token)
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    output = generate_with_truth_value(
        model=model,
        messages=None,
        question = question,
        truth_methods=truth_methods,
        max_new_tokens=64,
        temperature=0.7,
        batch_generation=True,
        eos_token_id=2,
        pad_token_id = 32001,
        processor = processor,
        input = inputs
    )
    return output

def load_dataset():
    df = pd.read_csv("data/popqa.tsv", sep="\t")
    return df



###The following are all tmp test functions


def test_model(model, processor, image_path="test.jpg", question="What is in this picture?"):

    image = Image.open(image_path).convert("RGB")

    # Prepare prompt (LLaVA expects <image> token)
    prompt = f"USER: <image>\n{question}\nASSISTANT:"

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    ).to(model.device)

    

    with torch.no_grad():
       output_ids = model.generate(
           **inputs,
           num_return_sequences=5,
           do_sample=True,
           return_dict_in_generate=True,
           output_attentions=False,
           output_hidden_states=False,
           output_logits=False,
           eos_token_id=2,
           pad_token_id=32001,
           #**kwargs
       )
    

    sequences = output_ids.sequences  

    # Decode each generated sequence into text
    decoded = processor.tokenizer.batch_decode(
        sequences, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )

    print(decoded)

    #with torch.no_grad():
    #    output_ids = model.generate(**inputs,  max_new_tokens=64)
    #print(output_ids)


        # Print tokens side by side
    print("Generated tokens (ID → symbol):")
    for tok_id in output_ids[0]:
        token_str = processor.tokenizer.decode([tok_id.item()], skip_special_tokens=False)
        print(f"{tok_id.item():<6} → '{token_str}'")
    answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0] 
    return answer
    #print(f"Q: {question}\nA: {answer}")

def ask_text_question(model, processor, question="What is the capital of France?"):
    # Prepare prompt (LLaVA-style format)
    
    prompt = f"USER: {question}\nASSISTANT:"

    # Encode text-only input
    inputs = processor(
        text=prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=64)

    # Decode full answer
    answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    #print(f"Q: {question}\nA: {answer}\n")

    # Print tokens side by side
    print("Generated tokens (ID → symbol):")
    for tok_id in output_ids[0]:
        token_str = processor.tokenizer.decode([tok_id.item()], skip_special_tokens=False)
        print(f"{tok_id.item():<6} → '{token_str}'")

    return answer, output_ids


model, processor = load_model()
truth_methods = load_truth_methods()

print(ask_question("What is in this picture?", "test.jpg", model, processor, truth_methods))

#print(test_model(model, processor))
#test_model(model, processor)
#test_model(model, processor)

#output  = ask_question("What is the capital of France?", model, processor, truth_methods)
#print(output)


#print(result)
#ask_text_question(model, processor)
#test_model(model, processor, image_path="test.jpg", question="Describe the main object in this image.")


