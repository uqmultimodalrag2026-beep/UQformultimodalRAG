from .truth_method import TruthMethod
from typing import Union
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

import torch
import numpy as np


class BM25_score(TruthMethod):
    def __init__(self, **kwargs):  
        super().__init__()

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
        **kwargs
    ):
        score = kwargs.pop("bm25_score")
        if type(BM25_score) == list:
            score = score[0]

        return {
            "truth_value": score,
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
        score = kwargs.pop("bm25_score")
        if type(BM25_score) == list:
            score = score[0]

        return {
            "truth_value": score,
        }  

