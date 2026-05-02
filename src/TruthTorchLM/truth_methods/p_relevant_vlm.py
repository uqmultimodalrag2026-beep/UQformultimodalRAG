from .truth_method import TruthMethod
from TruthTorchLM.utils import find_token_indices, fix_tokenizer_chat
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLM.templates import (
    PRELEVANT_SYSTEM_PROMPT,
    PRELEVANT_PROMPT,
    PRELEVANT_MODEL_OUTPUT,
)
from ..generation import sample_generations_hf_local, sample_generations_api

import torch
import numpy as np


class PRelevant_VLM(TruthMethod):


    def __init__(
        self,
        system_prompt: str = PRELEVANT_SYSTEM_PROMPT,
        user_prompt: str = PRELEVANT_PROMPT,
        model_output: str = PRELEVANT_MODEL_OUTPUT,
        batch_generation=True,
        with_context: bool = True,
    ):
        super().__init__()
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.model_output = model_output
        self.batch_generation = batch_generation
        self.with_context = with_context
        if with_context and '{context}' not in self.user_prompt:#check if the prompt has a context field
            print("Context field is required.")
            raise RuntimeError

    def forward_hf_local(
        self,
        model: PreTrainedModel,
        question: str,
        context: str = "",
        image = None,
        processor = None,
        **kwargs
    ):

        if "llava" in model.config._name_or_path:
            prompt = f"""SYSTEM: {self.system_prompt}
                USER: <image>
                {self.user_prompt.format(
                    question=question,
                    context=context,
                )}
                ASSISTANT: {self.model_output}"""

            prompt_tokens = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(**prompt_tokens)
                logits = outputs.logits  # Logits for each token in the input
            

            logprobs = torch.log_softmax(logits, dim=-1)  # logprobs for each token
            # logprobs for each token except the last one
            logprobs = logprobs[0, :-1, :]
            prompt_tokens = prompt_tokens["input_ids"]

            logprobs = torch.gather(
                logprobs, dim=1, index=prompt_tokens[0][1:].view(-1, 1)
            )  # logprobs for each token in the generated text
            
            logprobs = logprobs.view(-1).tolist()  # convert to list
            
            #initial implementation does not find the correct index of true for some reason, so we do it like this instead
            loss_true = logprobs[-1]
            if processor.tokenizer.decode(prompt_tokens[0][-1]) != "relevant":
                print("""Last token is not "relevant". Instead it is: """, processor.tokenizer.decode(prompt_tokens[0][-1]))

            prob_relevant = np.exp(loss_true).item()
            
            return {
                "truth_value": prob_relevant,
                "p_relevant": prob_relevant,
            }  # this output format should be same for all truth methods
        elif "Qwen" in model.config._name_or_path:
            # Build messages with system and user roles
            tokenizer=processor.tokenizer
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},  # optional image
                        {
                            "type": "text",
                            "text": self.user_prompt.format(
                                question=question,
                                context=context
                            ) + f"\nassistant\n{self.model_output}"
                        },
                    ],
                },
            ]

            # Fix tokenizer chat template if needed
            tokenizer, messages = fix_tokenizer_chat(tokenizer, messages)

            # Convert messages to single prompt string (tokenize=False)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False)

            # Encode prompt into input_ids tensor
            prompt_tokens = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model(prompt_tokens)
                logits = outputs.logits  # [1, seq_len, vocab_size]

            # Compute log-probs for all tokens except the last one
            logprobs = torch.log_softmax(logits, dim=-1)
            token_logprobs = torch.gather(
                logprobs[0, :-1, :],
                dim=1,
                index=prompt_tokens[0][1:].view(-1, 1)
            ).view(-1)

            input_ids_list = prompt_tokens[0][1:].tolist()
            token_logprobs_list = token_logprobs.tolist()

            # Find token indices for "relevant"
            indices, texts = find_token_indices(prompt_tokens[0][1:], tokenizer, "relevant")
            relevant_indices = indices[-1]  # last occurrence
            """
            # Print all tokens token by token, highlighting "relevant"
            print("\n--- Prompt tokens ---")
            for i, token_id in enumerate(input_ids_list):
                token_str = tokenizer.decode([token_id])
                lp = token_logprobs_list[i]
                if i in relevant_indices:
                    print(f"--> {token_str:<12} | logprob: {lp:.6f}   <-- evaluated token")
                else:
                    print(f"    {token_str:<12} | logprob: {lp:.6f}")
            """
            # Compute probability for "relevant" token(s)
            loss_relevant = sum(token_logprobs_list[i] for i in relevant_indices) / len(relevant_indices)
            prob_relevant = np.exp(loss_relevant).item()

            return {
                "truth_value": prob_relevant,
                "p_relevant": prob_relevant,
            }  # same output format for all truth methods
                

    def forward_api(
        self,
        model: str,
        messages: list,
        generated_text: str,
        question: str,
        generation_seed=None,
        sampled_generations_dict: dict = None,
        logprobs: list = None,
        generated_tokens: list = None,
        context: str = "",
        **kwargs
    ):
        raise NotImplementedError
