from .truth_method import TruthMethod
from TruthTorchLM.utils import find_token_indices, fix_tokenizer_chat
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from TruthTorchLM.templates import (
    PTRUE_SYSTEM_PROMPT,
    PTRUE_USER_PROMPT,
    PTRUE_MODEL_OUTPUT,
    PTRUE_USER_PROMPT_WITH_CONTEXT,
)
from ..generation import sample_generations_hf_local, sample_generations_api

import torch
import numpy as np


class PTrue_VLM(TruthMethod):

    REQUIRES_SAMPLED_TEXT = True

    def __init__(
        self,
        number_of_ideas: int = 5,
        system_prompt: str = PTRUE_SYSTEM_PROMPT,
        user_prompt: str = PTRUE_USER_PROMPT,
        model_output: str = PTRUE_MODEL_OUTPUT,
        batch_generation=True,
        with_context: bool = False,
        image  = None,
        processor = None
    ):
        super().__init__()
        self.number_of_ideas = number_of_ideas
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.model_output = model_output
        self.batch_generation = batch_generation
        self.with_context = with_context
        if with_context and '{context}' not in self.user_prompt:#check if the prompt has a context field
            print("Context field is required in user prompt for with_context=True, swithing to the default user prompt with context")
            self.user_prompt = PTRUE_USER_PROMPT_WITH_CONTEXT

    def forward_hf_local(
        self,
        model: PreTrainedModel,
        input_text: str,
        generated_text: str,
        question: str,
        all_ids: Union[list, torch.Tensor],
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast] = None,
        generation_seed=None,
        sampled_generations_dict: dict = None,
        messages: list = [],
        context: str = "",
        image = None,
        processor = None,
        **kwargs
    ):

        if sampled_generations_dict is None:
            kwargs.pop("input")
            sampled_generations_dict = sample_generations_hf_local(
                model=model,
                input_text=input_text,
                tokenizer=tokenizer,
                generation_seed=generation_seed,
                number_of_generations=self.number_of_ideas,
                return_text=True,
                batch_generation=self.batch_generation,
                **kwargs
            )

        generated_text = tokenizer.decode(
            tokenizer.encode(
                generated_text, return_tensors="pt").view(-1).tolist(),
            skip_special_tokens=True,
        )  # remove special tokens
        ideas = sampled_generations_dict["generated_texts"][: self.number_of_ideas]
        ideas = "\n".join(ideas)

        if "llava" in model.config._name_or_path:
            if self.with_context == False:
                prompt = f"""SYSTEM: {PTRUE_SYSTEM_PROMPT}
                    USER: <image>
                    {PTRUE_USER_PROMPT.format(
                        question=question,
                        ideas=ideas,
                        generated_text=generated_text
                    )}
                    ASSISTANT: {PTRUE_MODEL_OUTPUT}"""
            else:
                prompt = f"""SYSTEM: {PTRUE_SYSTEM_PROMPT}
                    USER: <image>
                    {PTRUE_USER_PROMPT_WITH_CONTEXT.format(
                        context = context,
                        question=question,
                        ideas=ideas,
                        generated_text=generated_text
                    )}
                    ASSISTANT: {PTRUE_MODEL_OUTPUT}"""

        elif "Qwen" in model.config._name_or_path:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": PTRUE_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": PTRUE_USER_PROMPT.format(
                                question=question,
                                ideas=ideas,
                                generated_text=generated_text,
                                **({"context": context} if self.with_context else {})
                            ),
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": PTRUE_MODEL_OUTPUT}],
                },
            ]
        if "llava" in model.config._name_or_path:
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
            if processor.tokenizer.decode(prompt_tokens[0][-1]) != "true":
                print("Last token is not true. Instead it is: ", processor.tokenizer.decode(prompt_tokens[0][-1]))

            prob_true = np.exp(loss_true).item()

            return {
                "truth_value": prob_true,
                "p_true": prob_true,
                "generated_ideas": ideas,
            }  # this output format should be same for all truth methods
        elif "Qwen" in model.config._name_or_path:
            # 1. Build model inputs correctly (THIS is what goes into the model)
            prompt_tokens = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**prompt_tokens)
                logits = outputs.logits  # [1, seq_len, vocab_size]

            # 2. Compute log-probs
            logprobs = torch.log_softmax(logits, dim=-1)

            # log p(token_t | prefix) aligns logits[t-1] with input_ids[t]
            input_ids = prompt_tokens["input_ids"][0]
            token_logprobs = torch.gather(
                logprobs[0, :-1, :],
                dim=1,
                index=input_ids[1:].unsqueeze(-1),
            ).squeeze(-1)

            token_logprobs_list = token_logprobs.tolist()
            input_ids_list = input_ids[1:].tolist()

            tokenizer = processor.tokenizer

            # 3. Find indices for the *last occurrence* of "true"
            indices, texts = find_token_indices(input_ids[1:], tokenizer, "true")
            true_indices = indices[-1]

            # 4. Debug print: token-by-token with highlight
            """
            print("\n--- Qwen prompt tokens ---")
            for i, token_id in enumerate(input_ids_list):
                token_str = tokenizer.decode([token_id])
                lp = token_logprobs_list[i]
                if i in true_indices:
                    print(f"--> {token_str:<12} | logprob: {lp:.6f}   <-- evaluated")
                else:
                    print(f"    {token_str:<12} | logprob: {lp:.6f}")
            """
            # 5. Compute probability for "true"
            loss_true = sum(token_logprobs_list[i] for i in true_indices) / len(true_indices)
            prob_true = float(np.exp(loss_true))

            return {
                "truth_value": prob_true,
                "p_true": prob_true,
                "generated_ideas": ideas,
            }




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
        # make sampling for the ideas
        if sampled_generations_dict is None:
            sampled_generations_dict = sample_generations_api(
                model=model,
                messages=messages,
                generation_seed=generation_seed,
                number_of_generations=self.number_of_ideas,
                return_text=True,
                **kwargs
            )

        ideas = sampled_generations_dict["generated_texts"][: self.number_of_ideas]
        ideas = "\n".join(ideas)
    
        if self.with_context == False:
            chat = [
                {"role": "system", "content": self.system_prompt},
                {
                "role": "user",
                "content": self.user_prompt.format(
                    question=question,
                    ideas=ideas,
                    generated_text=generated_text,
                ),
            },
        ]
        else:
            chat = [
                {"role": "system", "content": self.system_prompt},
                {
                "role": "user",
                "content": self.user_prompt.format(
                    question=question,
                    ideas=ideas,
                    generated_text=generated_text,
                    context=context,
                ),
            },
        ]

        sampled_generations_dict = sample_generations_api(
            model=model,
            messages=chat,
            generation_seed=generation_seed,
            number_of_generations=1,
            return_text=True,
            return_logprobs=True,
            temperature=0.0,
        )
        logprobs = sampled_generations_dict["logprobs"][0]
        tokens = sampled_generations_dict["tokens"][0]

        for i, token in enumerate(tokens):
            if "true" in token.lower():
                prob = np.exp(logprobs[i]).item()
                return {"truth_value": prob, "p_true": prob, "generated_ideas": ideas}
            if "false" in token.lower():
                prob = 1 - np.exp(logprobs[i]).item()
                return {"truth_value": prob, "p_true": prob, "generated_ideas": ideas}

        return {"truth_value": 0.5, "p_true": 0.5, "generated_ideas": ideas}
