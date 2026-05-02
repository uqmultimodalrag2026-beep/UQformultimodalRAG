import copy
import torch
import numpy as np
from tqdm import tqdm
from typing import Union
from datasets import Dataset
from sklearn.model_selection import train_test_split

from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from transformers import DebertaForSequenceClassification, DebertaTokenizer
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from .truth_method import TruthMethod
from TruthTorchLM.utils import bidirectional_entailment_clustering
from TruthTorchLM.templates import DEFAULT_SYSTEM_BENCHMARK_PROMPT, DEFAULT_USER_PROMPT, DEFAULT_LLAVA_PROMPT_NO_CONTEXT, DEFAULT_LLAVA_PROMPT_WITH_CONTEXT
from .semantic_entropy import calculate_total_log

from ..evaluators.correctness_evaluator import CorrectnessEvaluator
from TruthTorchLM.utils.dataset_utils import get_dataset
from ..generation_vlm import (
    sample_generations_hf_local,
    sample_generations_api,
    sample_generations_batch_hf_local,
    sample_generations_sequential_hf_local,
)
from TruthTorchLM.utils.eval_utils import metric_score
from TruthTorchLM.utils.common_utils import fix_tokenizer_chat

from TruthTorchLM.error_handler import handle_logprobs_error
from TruthTorchLM.utils.common_utils import generate, fix_tokenizer_chat, split_after_subarray
import math
import os
import pickle
import json

class LARS_BASE(TruthMethod):

    REQUIRES_LOGPROBS = True
    REQUIRES_SAMPLED_TEXT = True
    REQUIRES_SAMPLED_LOGPROBS = True
    lars_with_context = False

    def __init__(
        self,
        device="cuda",
        lars_model: PreTrainedModel = None,
        lars_tokenizer: PreTrainedTokenizer = None,
        ue_type: str = "confidence",
        number_of_generations: int = 0,
        model_for_entailment: PreTrainedModel = None,
        tokenizer_for_entailment: PreTrainedTokenizer = None,
        entailment_model_device="cuda",
        batch_generation:bool=True, #used only if ue_type is se or entropy
        lars_with_context = False #used to determine whether the whole context or just the question is given to the lars model
    ):
        super().__init__()
        self.lars_with_context = lars_with_context

        assert ue_type in [
            "confidence",
            "semantic_entropy",
            "se",
            "entropy",
        ], f"ue_type must be one of ['confidence', 'semantic_entropy', 'se', 'entropy'] but it is {ue_type}."
        self.ue_type = ue_type
        # number of generations for semantic entropy and entropy
        self.number_of_generations = number_of_generations
        self.batch_generation = batch_generation

        # lars model
        if lars_model is None or lars_tokenizer is None:
            lars_model = AutoModelForSequenceClassification.from_pretrained(
                "duygunuryldz/LARS"
            ).to(
                device
            )  # TODO
            lars_tokenizer = AutoTokenizer.from_pretrained(
                "duygunuryldz/LARS")  # TODO
        self.lars_model = lars_model
        self.lars_tokenizer = lars_tokenizer
        self.device = device

        # lars params
        self.number_of_bins = (
            lars_model.config.number_of_bins
        )  # number of bins for discretization of the probability space
        self.edges = (
            lars_model.config.edges
        )  # edges of bins, discretization of the probability space

        # params for semantic entropy
        if (ue_type == "se" or ue_type == "semantic_entropy") and (
            model_for_entailment is None or tokenizer_for_entailment is None
        ):
            model_for_entailment = DebertaForSequenceClassification.from_pretrained(
                "microsoft/deberta-large-mnli"
            ).to(entailment_model_device)
            tokenizer_for_entailment = DebertaTokenizer.from_pretrained(
                "microsoft/deberta-large-mnli"
            )
            assert self.number_of_generations > 0, "Number of generations should be bigger that 0 if UE type is SE or Entropy"

        self.model_for_entailment = model_for_entailment
        self.tokenizer_for_entailment = tokenizer_for_entailment

    @staticmethod
    def _find_bin(value, edges, number_of_bins):
        if edges is not None:
            bin_index = np.digitize(value, edges, right=False)
        else:
            bin_index = int(
                value * number_of_bins
            )  # discretize the probability space equally
        return min(bin_index, (number_of_bins - 1))

    @staticmethod
    def prepare_answer_text(probs, answer_tokens, edges, number_of_bins):
        a_text = ""
        assert len(probs) == len(answer_tokens)
        for i, tkn_text in enumerate(answer_tokens):
            bin_id = LARS_BASE._find_bin(probs[i], edges, number_of_bins)
            a_text += tkn_text + f"[prob_token_{bin_id}]"
        return a_text

    @staticmethod
    def tokenize_input(tokenizer, question, answer_text):

        tokenized_input = tokenizer(
            question,
            answer_text,
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            return_token_type_ids=True,
            is_split_into_words=False,  # ???
            truncation=True,
            max_length=None,
            padding="max_length",
        )
        return tokenized_input
    
    def _identify_token_spans(
    self,
    tokens,
    question,
    probs,
    instruction_length,
    changed_eot,
):
        """
        Identifies token index spans for:
        CLS, context, question, instruction, answer, padding, separators
        """

        # --- tokenize question alone ---
        q_enc = self.lars_tokenizer(
            question,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        question_ids = q_enc["input_ids"]

        cls_token = self.lars_tokenizer.cls_token      # <s>
        sep_token = self.lars_tokenizer.sep_token      # </s>
        pad_token = self.lars_tokenizer.pad_token      # <pad>

        cls_idx = None
        sep_indices = []
        pad_indices = []

        for i, tok in enumerate(tokens):
            if tok == cls_token and cls_idx is None:
                cls_idx = i
            if tok == sep_token:
                sep_indices.append(i)
            if tok == pad_token:
                pad_indices.append(i)

        spans = {
            "cls_idx": cls_idx,
            "sep_indices": sep_indices,
            "pad_start": pad_indices[0] if pad_indices else None,
            "context_start": None,
            "context_end": None,
            "question_start": None,
            "question_end": None,
            "instruction_start": None,
            "instruction_end": None,
            "answer_start": None,
            "answer_end": None,
        }

        # Expected RoBERTa layout:
        # <s> context question instruction </s> </s> answer </s> <pad> ...

        if len(sep_indices) >= 3:
            context_question_start = cls_idx + 1
            context_question_end = sep_indices[0] - instruction_length + 1

            context_end = context_question_end - len(question_ids)
            question_start = context_end
            question_end = context_question_end

            instruction_start = question_end
            instruction_end = sep_indices[0]

            answer_start = sep_indices[1] + 1
            if changed_eot:
                answer_end = answer_start + len(probs) * 2 + 7
            else:
                answer_end = answer_start + len(probs) * 2

            spans.update({
                "context_start": context_question_start,
                "context_end": context_end,
                "question_start": question_start,
                "question_end": question_end,
                "instruction_start": instruction_start,
                "instruction_end": instruction_end,
                "answer_start": answer_start,
                "answer_end": answer_end,
            })

        return spans
    
    

    def _normalized_entropy(self, values):
        """
        values: list or 1D tensor of attention values for a class
        returns normalized entropy in [0, 1]
        """
        if len(values) <= 1:
            return 0.0

        eps = 1e-12
        total = sum(values)

        if total <= eps:
            return 0.0

        probs = [(v / total) + eps for v in values]
        entropy = -sum(p * math.log(p) for p in probs)
        return entropy / math.log(len(values))
    
    def _compute_cls_attention(self, output, spans, pkl_file=None, print_output=True):
        """
        Computes CLS → token attention (last layer),
        aggregates by class, and either prints or writes to a pickle file.

        Additionally computes:
            - number of tokens per class
            - average attention per token per class

        Args:
            output: model output with attentions
            spans: dict of token spans
            pkl_file: str, optional path to .pkl file to store results
            print_output: bool, whether to print the results
        """
        assert not (pkl_file is not None and print_output is False) or not (pkl_file is not None and print_output), \
            "Only one of printing or writing to pkl should be used."

        attentions = output.attentions  # tuple of layers
        num_layers = len(attentions)

        all_layers_data = []  # used only if saving to pkl

        for layer_idx, layer_attn in enumerate(attentions):
            cls_attn = layer_attn[0, :, spans["cls_idx"], :]  # (num_heads, seq_len)
            cls_attn_mean = cls_attn.mean(dim=0)  # average over heads → (seq_len,)

            # collect raw values per class
            attn_values = {
                "CLS": [],
                "CTX": [],
                "Q": [],
                "INSTRCT": [],
                "ANS": [],
                "SEP": [],
                "PAD": [],
            }

            for i, score in enumerate(cls_attn_mean.tolist()):
                if i == spans["cls_idx"]:
                    attn_values["CLS"].append(score)
                elif spans["context_start"] <= i < spans["context_end"]:
                    attn_values["CTX"].append(score)
                elif spans["question_start"] <= i < spans["question_end"]:
                    attn_values["Q"].append(score)
                elif spans["instruction_start"] <= i < spans["instruction_end"]:
                    attn_values["INSTRCT"].append(score)
                elif spans["answer_start"] <= i < spans["answer_end"]:
                    attn_values["ANS"].append(score)
                elif i in spans["sep_indices"]:
                    attn_values["SEP"].append(score)
                elif spans["pad_start"] is not None and i >= spans["pad_start"]:
                    attn_values["PAD"].append(score)

            # Aggregate sum, count, average, and entropy per class
            layer_data = {}
            total_mass = 0.0
            total_tokens = 0

            for cls, values in attn_values.items():
                cls_sum = sum(values)
                cls_count = len(values)
                cls_avg = cls_sum / cls_count if cls_count > 0 else 0.0
                cls_entropy = self._normalized_entropy(values)
                total_mass += cls_sum
                total_tokens += cls_count

                layer_data[cls] = {
                    "sum": cls_sum,
                    "count": cls_count,
                    "avg": cls_avg,
                    "entropy": cls_entropy,
                }

            layer_data["TOTAL"] = {
                "sum": total_mass,
                "count": total_tokens,
                "avg": total_mass / total_tokens if total_tokens > 0 else 0.0,
                "entropy": None,
            }

            all_layers_data.append(layer_data)

            # Print if requested
            if print_output:
                print(f"\n=== CLS → TOKEN CLASS ATTENTION (Layer {layer_idx+1}/{num_layers}) ===")
                print(f"{'CLASS':8s} {'SUM':>10s} {'COUNT':>8s} {'AVG':>10s} {'ENTROPY':>10s}")
                for cls, metrics in layer_data.items():
                    entropy = metrics["entropy"] if metrics["entropy"] is not None else 0.0
                    print(
                        f"{cls:8s} "
                        f"{metrics['sum']:10.4f} "
                        f"{metrics['count']:8d} "
                        f"{metrics['avg']:10.4f} "
                        f"{entropy:10.4f}"
                    )

        # Save to pickle if requested
        if pkl_file is not None:
            # Load existing data if the file exists
            if os.path.exists(pkl_file):
                with open(pkl_file, "rb") as f:
                    existing_data = pickle.load(f)
            else:
                existing_data = []

            existing_data.append(all_layers_data)

            with open(pkl_file, "wb") as f:
                pickle.dump(existing_data, f)

    #for debugging
    def _print_tokens_with_class(self, tokens, spans, input_ids_1d=None):
        """
        Print each token with its associated class label.
        """
        if input_ids_1d is None:
            input_ids_1d = [self.lars_tokenizer.convert_tokens_to_ids(tok) for tok in tokens]

        print("\n=== TOKENS WITH CLASS LABELS ===")
        print(f"{'IDX':>3s} | {'TOKEN':>12s} | {'ID':>6s} | CLASS")
        print("-" * 40)

        for i, (tok, tok_id) in enumerate(zip(tokens, input_ids_1d)):
            label = ""

            if i == spans["cls_idx"]:
                label = "CLS"
            elif spans["answer_start"] is not None and spans["answer_start"] <= i < spans["answer_end"]:
                label = "ANS"
            elif i in spans["sep_indices"]:
                label = "SEP"
            elif spans["pad_start"] is not None and i >= spans["pad_start"]:
                label = "PAD"
            elif spans["context_start"] is not None:
                if spans["context_start"] <= i < spans["context_end"]:
                    label = "CTX"
                elif spans["question_start"] <= i < spans["question_end"]:
                    label = "Q"
                elif spans["instruction_start"] <= i < spans["instruction_end"]:
                    label = "INSTRCT"

            print(f"{i:3d} | {tok:>12s} | {tok_id:6d} | {label}")


    

    def _lars(self, question, full_text, generation_token_texts, probs, instruction_length = 12, change_to_eot = False, comp_att = False):
        #Test: Replace the last EOS of generation tokens texts with eot token of roberta?. No need I think, no significant change
        #print(generation_token_texts, probs)
        changed = False
        if change_to_eot:
            for i, tok in enumerate(generation_token_texts):
                if tok == '</s>':
                    generation_token_texts[i] = '<|eot_id|>'
                    changed = True
        #Test end
        a_text = LARS_BASE.prepare_answer_text(
            probs, generation_token_texts, self.edges, self.number_of_bins
        )
        tokenized_input = LARS_BASE.tokenize_input(
            self.lars_tokenizer, full_text, a_text)
        input_ids = (
            torch.tensor(tokenized_input["input_ids"]).reshape(
                1, -1).to(self.device)
        )
        """
        decoded_input = self.lars_tokenizer.decode(
            input_ids[0],
            skip_special_tokens=False
        )
        print(decoded_input)
        """
        input_ids_1d = input_ids[0].tolist()
        tokens = self.lars_tokenizer.convert_ids_to_tokens(input_ids_1d)

        if comp_att:
            spans = self._identify_token_spans(
                tokens=tokens,
                question=question,
                probs=probs,
                instruction_length=instruction_length,
                changed_eot = changed
            )

        #self._print_tokens_with_class(tokens, spans, input_ids_1d=input_ids_1d)
           
        attention_mask = (
            torch.tensor(tokenized_input["attention_mask"])
            .reshape(1, -1)
            .to(self.device)
        )
        token_type_ids = (
            torch.tensor(tokenized_input["token_type_ids"])
            .reshape(1, -1)
            .to(self.device)
        )
        with torch.no_grad():
            self.lars_model.eval()
            output = self.lars_model(
                input_ids, attention_mask=attention_mask, 
                token_type_ids=token_type_ids,
                output_attentions=True,
                return_dict=True
            )
        logits = output.logits.detach()

        if comp_att:
            self._compute_cls_attention(output, spans, print_output=False, pkl_file="attentions_context.pkl")

        output = torch.nn.functional.sigmoid(logits[:, 0]).item()
        #print(output) #<-- scalar
        return output

    @staticmethod
    def extract_between(text, start, end):
        start_idx = text.index(start) + len(start)
        end_idx = text.index(end, start_idx)
        return text[start_idx:end_idx]
    
    def _infer_probs_qwen(self, model, processor, all_ids, input_text, generated_text, image, forward_hf = True):
        tokenizer = processor.tokenizer
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(
                model.device
                )
        model_output = all_ids.to(model.device)
        tokens = model_output[0][len(input_ids[0]):]
        question_text = tokenizer.batch_decode(input_ids)
        question_and_context = self.extract_between(question_text[0], "<|vision_end|>", "<|im_end|>") #everything between vision end and im_end
        messages_prompt_only = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question_and_context},
                ],
            }
        ]
        messages_with_answer = [
            *messages_prompt_only,
            {
                "role": "assistant",
                "content": [{"type": "text", "text": generated_text}],
            },
        ]

        # --- 2. Encode prompt-only ---
        prompt_only_tokens = processor.apply_chat_template(
            messages_prompt_only,
            tokenize=True,
            add_generation_prompt=True,  # assistant starts here
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)


        # --- 3. Encode prompt + answer ---
        full_tokens = processor.apply_chat_template(
            messages_with_answer,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        # --- 4. Forward pass ---
        with torch.no_grad():
            outputs = model(**full_tokens)
            logits = outputs.logits  # [1, seq_len, vocab]

        # --- 5. Compute where the answer starts ---
        len_input = prompt_only_tokens["input_ids"].shape[1]

        # --- 6. Extract probabilities for answer tokens ---
        probs = torch.softmax(logits, dim=-1)

        answer_token_ids = full_tokens["input_ids"][0][len_input:] 
        answer_probs = probs[0, len_input - 1 :, :] 
        
        token_probs = torch.gather(
            answer_probs,
            dim=1,
            index=answer_token_ids.unsqueeze(-1),
        ).squeeze(-1)

        probs_list = token_probs.tolist()

        # --- 7. Decode tokens (for LARS) ---
        tokens_text = processor.tokenizer.batch_decode(
            answer_token_ids, skip_special_tokens=False
        )
        return probs_list, tokens_text, question_and_context
    
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
        processor = None,
        image = None,
        messages: list = [],
        context: str = "",
        **kwargs,
    ):
        #toDo: update larsVLM accordingly
        if self.ue_type == "confidence":
            model_output = all_ids.to(model.device) 
            if "llava" in model.config._name_or_path:
                target = torch.tensor([22933, 9047, 13566, 29901]).to("cuda")
                tokens = split_after_subarray(output_ids=model_output , target=target) 
                tokens_text = [processor.tokenizer.decode([token]) for token in tokens]
                full_text = processor.batch_decode(model_output, skip_special_tokens=False)[0].split("\n")[1:]
                full_text = " ".join(full_text)
                prompt = f"USER: <image>\n {full_text}"
                generated_output = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model(**generated_output) 
                    logits = outputs.logits  # (1, seq_len, vocab)
                    len_input = logits[0].shape[0] - tokens.shape[0]
                
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    probs = probs[0, len_input-1:, :]
                    probs = torch.gather(
                        probs, dim=1, index=generated_output['input_ids'][0][len_input:].view(-1, 1) 
                    ) 
                    probs = probs.view(-1).tolist()
                    #only answer
                    #lars_score = self._lars("", "", tokens_text, probs)
                    if self.lars_with_context:
                        lars_score = self._lars(question, full_text.split(" ASSISTANT")[0], tokens_text, probs) #with context
                    else:
                        lars_score = self._lars(question, question, tokens_text, probs) #without context
            elif "Qwen" in model.config._name_or_path:

                probs, tokens_text, full_text = self._infer_probs_qwen(model, processor, all_ids, input_text, generated_text, image) #model, processor, all_ids, input_text, generated_text, image,
                with torch.no_grad():
                    if self.lars_with_context:
                        lars_score = self._lars(question, full_text, tokens_text, probs) #with context
                    else:
                        lars_score = self._lars(question, question, tokens_text, probs) #without context

        else:
            raise RuntimeError("Only Confidence is allowed!")

        return {
            "truth_value": lars_score,
            "generated_text": generated_text,
        }  # we shouldn't return generated text. remove it from the output format
    
    #For computation of LARS score in not main exp line.
    def lars_postcompute(self, model, processor, prompt, question,image, context, output):
        prompt = prompt.format(context=context, question = question) +" " + output+ "</s>"
        full_text = prompt.split("\n")[1:]
        full_text = " ".join(full_text)
        prompt = f"USER: <image>\n {full_text}"
        generated_output = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

        target = torch.tensor([319, 1799, 9047, 13566, 29901]).to("cuda")
        tokens = split_after_subarray(output_ids=generated_output.input_ids , target=target)
        tokens_text = [processor.tokenizer.decode([token]) for token in tokens] 
        with torch.no_grad():
                outputs = model(**generated_output) 
                logits = outputs.logits  # (1, seq_len, vocab)
                #print(logits)
                len_input = logits[0].shape[0] - tokens.shape[0]
                probs = torch.nn.functional.softmax(logits, dim=-1)
                probs = probs[0, len_input-1:, :]
                probs = torch.gather(
                    probs, dim=1, index=generated_output['input_ids'][0][len_input:].view(-1, 1) 
                ) 
                probs = probs.view(-1).tolist()
                #print(generated_output.input_ids.tolist())
                #print(tokens_text)
                #print(probs)
                lars_score = self._lars(question, question, tokens_text, probs) #without context
        return lars_score

        


    @handle_logprobs_error
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
        **kwargs,
    ):

        if self.ue_type == "confidence":
            lars_score = self._lars(
                question, generated_tokens, torch.exp(
                    torch.tensor(logprobs))
            )

        elif self.ue_type in ["semantic_entropy", "se", "entropy"]:
            if sampled_generations_dict is None:
                sampled_generations_dict = sample_generations_api(
                    model=model,
                    messages=messages,
                    generation_seed=generation_seed,
                    number_of_generations=self.number_of_generations,
                    return_text=True,
                    return_logprobs=True,
                    **kwargs,
                )
            scores = []
            generated_outputs = []
            generated_texts = sampled_generations_dict["generated_texts"][
                : self.number_of_generations
            ]

            for i in range(self.number_of_generations):
                score = torch.log(
                    torch.tensor(self._lars(
                        question,
                        sampled_generations_dict["tokens"][i],
                        torch.exp(torch.tensor(sampled_generations_dict["logprobs"][i])),
                    ))
                ).item()
                scores.append(score)  # scores are in log scale
                generated_outputs.append((generated_texts[i], score))

            if self.ue_type == "semantic_entropy" or self.ue_type == "se":
                clusters = bidirectional_entailment_clustering(
                    self.model_for_entailment,
                    self.tokenizer_for_entailment,
                    question,
                    sampled_generations_dict["generated_texts"],
                )
                lars_score = -calculate_total_log(generated_outputs, clusters)
                return {
                    "truth_value": lars_score,
                    "score_for_each_generation": scores,
                    "generated_texts": generated_texts,
                    "clusters": clusters,
                }
            elif self.ue_type == "entropy":
                lars_score = np.sum(scores) / len(scores)
                return {
                    "truth_value": lars_score,
                    "score_for_each_generation": scores,
                    "generated_texts": generated_texts,
                }

        return {
            "truth_value": lars_score,
            "generated_text": generated_text,
        }  # we shouldn't return generated text. remove it from the output format

    @staticmethod
    def _get_datasets(
        datasets: list, size_for_each_dataset: list, val_ratio: float, seed: int, context_mode = None, dataset_csv=None, dataset_path=None, image_directory=None, retrieval_file=None, kb_path = None
    ):
        print("Creating train and validation datasets...")
        all_data = []
        for i, dataset in enumerate(datasets):
            all_data.append(
                get_dataset(
                    dataset,
                    size_of_data=size_for_each_dataset[i],
                    seed=seed,
                    split="train",
                    context_mode = context_mode,
                    dataset_csv=dataset_csv,
                    dataset_path=dataset_path,
                    kb_path=kb_path,
                    image_directory=image_directory,
                    retrieval_file = retrieval_file
                )
            )

        all_data = sum(
            all_data, []
        )  # list of dict, each dict contains "question" and "ground_truths"
        train_data, val_data = train_test_split(
            all_data, test_size=val_ratio, random_state=seed
        )
        return train_data, val_data

    @staticmethod
    def _generate_answers_and_label(
        train_data: list,
        val_data: list,
        model: PreTrainedModel,
        processor: AutoProcessor,
        correctness_evaluator: CorrectnessEvaluator,
        previous_context: list = [
            {"role": "system", "content": DEFAULT_SYSTEM_BENCHMARK_PROMPT}
        ],
        user_prompt: str = DEFAULT_LLAVA_PROMPT_NO_CONTEXT,
        num_gen_per_question: int = 5,
        **kwargs,
    ):
        text = ""
        print("Generating answers and labels for training data...")
        for i in tqdm(range(len(train_data))):
            question=train_data[i]["question"]
            context=train_data[i]["context"]
            image = train_data[i]["image"]
            
            if "{context}" in user_prompt:
                prompt = user_prompt.format(question=question, context = context)
            else: 
                prompt = user_prompt.format(question=question)
            if LARS_BASE.lars_with_context:
                full_text = prompt.split("\n")[1:]
                train_data[i]["question"] = "".join(full_text)
            if "Qwen" in model.config._name_or_path:
                messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
                ]
                input = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
                ).to(model.device)
                
            else:
                input = processor(
                text=prompt,
                images = image,
                return_tensors="pt"
                ).to(model.device)
            sampled = sample_generations_batch_hf_local(
                model=model,
                input=input,
                input_text=text,
                processor=processor,
                number_of_generations=num_gen_per_question - 1,
                return_text=True,
                return_logprobs=True,
                return_model_output=False,
                **kwargs,
            )

            most_likely = sample_generations_sequential_hf_local(
                model,
                input=input,
                input_text=text,
                tokenizer=processor.tokenizer,
                number_of_generations=1,
                do_sample=False,
                return_text=True,
                return_logprobs=True,
                return_model_output=False,
                top_p=1,
                temperature=None,
                **kwargs,
            )
            #due to the regeneration of the input, when using QWEN we also model the "\n" token followed by the eos token during inference. We append the "\n" token with a prob of 1.0 to make it consistent
            if "Qwen" in model.config._name_or_path: 
                train_data[i]["generated_texts"] = most_likely["generated_texts"] 
                train_data[i]["probs"] = [
                    np.exp(most_likely["logprobs"][0]).tolist()+ [1.0]] 
                train_data[i]["token_texts"] = [
                    [processor.decode(token) for token in most_likely["tokens"][0]] + ['\n']  
                ]            
                for j in range(len(sampled["generated_texts"])):
                    if sampled["generated_texts"][j] in train_data[i]["generated_texts"]:
                        continue
                    train_data[i]["generated_texts"].append(
                        sampled["generated_texts"][j])
                    train_data[i]["probs"].append(
                        np.exp(sampled["logprobs"][j]).tolist() +  [1.0])
                    train_data[i]["token_texts"].append(
                        [processor.decode([token])
                        for token in sampled["tokens"][j]]+ ['\n']
                    )
                train_data[i]["labels"] = [
                    correctness_evaluator(
                        train_data[i]["question"], answer, train_data[i]["ground_truths"]
                    )
                    for answer in train_data[i]["generated_texts"]
                ]
            else:
                train_data[i]["generated_texts"] = most_likely["generated_texts"]
                train_data[i]["probs"] = [
                    np.exp(most_likely["logprobs"][0]).tolist()]
                train_data[i]["token_texts"] = [
                    [processor.decode(token) for token in most_likely["tokens"][0]]
                ]
                #due to the regeneration of the input, when using QWEN we also model the "\n" token followed by the eos token during inference. We append the "\n" token with a prob of 1.0 to make it consistent
                
                for j in range(len(sampled["generated_texts"])):
                    if sampled["generated_texts"][j] in train_data[i]["generated_texts"]:
                        continue
                    train_data[i]["generated_texts"].append(
                        sampled["generated_texts"][j])
                    train_data[i]["probs"].append(
                        np.exp(sampled["logprobs"][j]).tolist())
                    train_data[i]["token_texts"].append(
                        [processor.decode([token])
                        for token in sampled["tokens"][j]]
                    )
                train_data[i]["labels"] = [
                    correctness_evaluator(
                        train_data[i]["question"], answer, train_data[i]["ground_truths"]
                    )
                    for answer in train_data[i]["generated_texts"]
                ]

        print("Generating answers and labels for validation data...")
        for i in tqdm(range(len(val_data))):
            question=val_data[i]["question"]
            context=val_data[i]["context"]
            image = val_data[i]["image"]
            if "{context}" in user_prompt:
                prompt = user_prompt.format(question=question, context = context)
            else: 
                prompt = user_prompt.format(question=question)
            if LARS_BASE.lars_with_context:
                full_text = prompt.split("\n")[1:]
                train_data[i]["question"] = "".join(full_text)
            if "Qwen" in model.config._name_or_path:
                messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
                ]
                input = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
                ).to(model.device)
            else:
                input = processor(
                text=prompt,
                images = image,
                return_tensors="pt"
                ).to(model.device)

            most_likely = sample_generations_sequential_hf_local(
                model,
                input = input,
                input_text=text,
                tokenizer=processor.tokenizer,
                number_of_generations=1,
                do_sample=False,
                return_text=True,
                return_logprobs=True,
                return_model_output=False,
                top_p=1,
                temperature=None,
                **kwargs,
            )
            if "Qwen":
                val_data[i]["generated_text"] = most_likely["generated_texts"][0]
                val_data[i]["probs"] = np.exp(most_likely["logprobs"][0]).tolist() + [1.0]
                val_data[i]["token_texts"] = [
                    processor.decode([token]) for token in most_likely["tokens"][0]
                ] + ['\n']
                val_data[i]["label"] = correctness_evaluator(
                    val_data[i]["question"],
                    most_likely["generated_texts"][0],
                    val_data[i]["ground_truths"],
                )
            else:
                val_data[i]["generated_text"] = most_likely["generated_texts"][0]
                val_data[i]["probs"] = np.exp(most_likely["logprobs"][0]).tolist()
                val_data[i]["token_texts"] = [
                    processor.decode([token]) for token in most_likely["tokens"][0]
                ]
                val_data[i]["label"] = correctness_evaluator(
                    val_data[i]["question"],
                    most_likely["generated_texts"][0],
                    val_data[i]["ground_truths"],
                )
        return train_data, val_data
    
    @staticmethod
    def save_datasets(train_data, val_data, save_path):
        os.makedirs(save_path, exist_ok=True)

        with open(os.path.join(save_path, "train_data.pkl"), "wb") as f:
            pickle.dump(train_data, f)

        with open(os.path.join(save_path, "val_data.pkl"), "wb") as f:
            pickle.dump(val_data, f)

    @staticmethod
    def load_datasets(save_path):
        with open(os.path.join(save_path, "train_data.pkl"), "rb") as f:
            train_data = pickle.load(f)

        with open(os.path.join(save_path, "val_data.pkl"), "rb") as f:
            val_data = pickle.load(f)

        return train_data, val_data
    
    @staticmethod
    def generate_and_save_labeled_data(
        datasets,
        size_for_each_dataset,
        val_ratio,
        seed,
        chat_model_name,
        correctness_evaluator,
        context_mode=None,
        dataset_csv=None,
        dataset_path=None,
        kb_path = None,
        image_directory=None,
        retrieval_file = None,
        previous_context=None,
        user_prompt=None,
        num_gen_per_question=5,
        save_data_path=None,
        **kwargs,
    ):
        assert val_ratio > 0

        # 1. Load and split datasets
        train_data, val_data = LARS_BASE._get_datasets(
            datasets,
            size_for_each_dataset,
            val_ratio,
            seed,
            context_mode=context_mode,
            dataset_csv=dataset_csv,
            dataset_path=dataset_path,
            kb_path = kb_path,
            image_directory=image_directory,
            retrieval_file = retrieval_file
        )

        # 2. Load model + processor (8-bit)
        if chat_model_name == "llava-hf/llava-1.5-7b-hf":
            quant_config = BitsAndBytesConfig(load_in_8bit=True)

            chat_processor = AutoProcessor.from_pretrained(chat_model_name)
            chat_model = LlavaForConditionalGeneration.from_pretrained(
                chat_model_name,
                quantization_config=quant_config,
                device_map="auto",
            )
        else:
            chat_model = Qwen3VLForConditionalGeneration.from_pretrained(
            chat_model_name,
            dtype="auto",           # automatically chooses FP16/FP32 depending on device
            device_map="auto"       # automatically places layers on GPU/CPU
        )
            chat_processor = AutoProcessor.from_pretrained(chat_model_name)

        # 3. Generate answers and labels
        train_data, val_data = LARS_BASE._generate_answers_and_label(
            train_data=train_data,
            val_data=val_data,
            model=chat_model,
            processor=chat_processor,
            correctness_evaluator=correctness_evaluator,
            previous_context=previous_context,
            user_prompt=user_prompt,
            num_gen_per_question=num_gen_per_question,
            pad_token_id=chat_processor.tokenizer.pad_token_id,
            eos_token_id=chat_processor.tokenizer.eos_token_id,
            **kwargs,
        )

        # 4. Cleanup
        del chat_model
        del chat_processor

        # 5. Save if requested
        if save_data_path:
            LARS_BASE.save_datasets(train_data, val_data, save_data_path)

        return train_data, val_data


    @staticmethod
    def _prepare_model_and_tokenizer(model, tokenizer, number_of_bins, edges):

        print("Preparing LARS model and tokenizer...")
        # add new prob tokens
        prob_tokens = [f"[prob_token_{i}]" for i in range(number_of_bins)]
        num_added_toks = tokenizer.add_special_tokens(
            {"additional_special_tokens": prob_tokens}
        )
        print("Number of tokens added:", num_added_toks)
        model.resize_token_embeddings(len(tokenizer))

        # initialize embeddings of prob tokens
        embeddings = model.get_input_embeddings()  #
        num_ones = int(embeddings.weight.data.shape[1] / number_of_bins)
        scale = (
            torch.sum(torch.abs(embeddings.weight.data))
            / embeddings.weight.data.shape[1]
            / embeddings.weight.data.shape[0]
        )
        with torch.no_grad():
            for i in range(number_of_bins):
                idx = number_of_bins - i - 1
                vector = torch.zeros(embeddings.weight.data[0].shape)
                vector[num_ones * idx: num_ones * (idx + 1)] = (
                    1.0 * scale * number_of_bins
                )
                embeddings.weight.data[-(i + 1)] = vector

        model.config.edges = list(edges)
        model.config.number_of_bins = number_of_bins

    @staticmethod
    def _prepare_data(
        tokenizer, train_data: list, val_data: list, number_of_bins: int, edges: list
    ):

        print("Preparing train data for LARS training...")
        all_data = []
        for d in tqdm(train_data):
            question = d["question"]
            for i in range(len(d["probs"])):
                if d["labels"][i] != -1:
                    ans_text = LARS_BASE.prepare_answer_text(
                        d["probs"][i], d["token_texts"][i], edges, number_of_bins
                    )
                    tokenized_input = LARS_BASE.tokenize_input(
                        tokenizer, question, ans_text)
                    all_data.append(
                        {
                            "label": d["labels"][i],
                            "input_ids": tokenized_input["input_ids"],
                            "token_type_ids": tokenized_input["token_type_ids"],
                            "attention_mask": tokenized_input["attention_mask"],
                        }
                    )
        print("Preparing validation data for LARS training...")
        all_test_data = []
        for d in tqdm(val_data):
            ans_text = LARS_BASE.prepare_answer_text(
                d["probs"], d["token_texts"], edges, number_of_bins
            )
            tokenized_input = LARS_BASE.tokenize_input(
                tokenizer, d["question"], ans_text)
            if d["label"] != -1:
                all_test_data.append(
                    {
                        "label": d["label"],
                        "input_ids": tokenized_input["input_ids"],
                        "token_type_ids": tokenized_input["token_type_ids"],
                        "attention_mask": tokenized_input["attention_mask"],
                    }
                )
        return Dataset.from_list(all_data), Dataset.from_list(all_test_data)

    @staticmethod
    def _test_loss(test_data, model, metrics, device):
        model.eval()
        losses = []
        scores = []
        labels = []
        cross_loss = torch.nn.BCEWithLogitsLoss()
        for i in range(len(test_data)):
            # test loss code
            label = torch.tensor(
                test_data[i]["label"]).reshape(1, -1).to(device)
            input_ids = (
                torch.tensor(test_data[i]["input_ids"]
                             ).reshape(1, -1).to(device)
            )
            attention_mask = (
                torch.tensor(test_data[i]["attention_mask"]).reshape(
                    1, -1).to(device)
            )
            token_type_ids = (
                torch.tensor(test_data[i]["token_type_ids"]).reshape(
                    1, -1).to(device)
            )

            logits = model(
                input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            ).logits.detach()
            loss = cross_loss(logits[:, :], label.reshape(-1, 1).float())
            scores.append(torch.nn.functional.sigmoid(logits[:, 0]).item())
            labels.append(label.item())
            losses.append(loss.item())

        losses = np.array(losses)
        metric_scores = metric_score(metrics, labels, scores, scores)

        return np.mean(losses), metric_scores

    @staticmethod
    def _train(
        train_dataset: Dataset,
        val_dataset: Dataset,
        model,
        tokenizer,
        save_path: str = None,
        device="cuda",
        test_metrics: list[str] = ["auroc"],
        main_metric: str = "auroc",
        wandb_run=None,
        number_of_bins: int = 8,
        epochs: int = 3,
        lr: float = 5e-6,
        batch_size: int = 8,
        test_freq: int = 100,
    ):

        expected_features = {"label", "input_ids",
                             "token_type_ids", "attention_mask"}
        assert expected_features.issubset(set(train_dataset.features.keys()))
        assert expected_features.issubset(set(val_dataset.features.keys()))
        assert main_metric in test_metrics

        print("LARS training started...")

        def custom_collate_fn(batch):
            # Convert lists to tensors and stack them
            labels = torch.stack([torch.tensor(item["label"])
                                 for item in batch])
            inps = torch.stack([torch.tensor(item["input_ids"])
                               for item in batch])
            types = torch.stack(
                [torch.tensor(item["token_type_ids"]) for item in batch]
            )
            masks = torch.stack(
                [torch.tensor(item["attention_mask"]) for item in batch]
            )

            return labels, inps, types, masks

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            pin_memory_device=device,
        )

        # Set loss and optimizer
        cross_loss = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        model.to(device)

        best_score = 0.0
        best_model = None

        with torch.no_grad():
            model.eval()
            tloss, metric_scores = LARS_BASE._test_loss(
                val_dataset, model, test_metrics, device
            )
            log = f"Test loss: {tloss:.2f}"
            for key, val in metric_scores.items():
                log += f"  | Test {key}: {val:.2f}"
            print(log)
            if wandb_run:
                wandb_run.log(
                    {"iter": 0, "test_loss": tloss}.update(metric_scores))
            model.train()

        for epoch in range(epochs):
            model.train()
            train_loss, total_sample = 0, 0
            for iteration, (
                labels,
                input_ids,
                token_type_ids,
                attention_mask,
            ) in enumerate(train_loader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                labels = labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                logits = model(
                    input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                ).logits
                loss = cross_loss(logits, labels.reshape(-1, 1).float())

                loss.backward()

                # mask the embeddings of the prob tokens -- only works for roberta
                grad_mask = torch.ones_like(
                    model.roberta.embeddings.word_embeddings.weight.grad
                )
                grad_mask[-number_of_bins:] = 0.0
                model.roberta.embeddings.word_embeddings.weight.grad *= grad_mask

                optimizer.step()
                train_loss += loss.item() * len(labels)
                total_sample += len(labels)

                if (iteration) % test_freq == 0:
                    model.eval()

                    with torch.no_grad():
                        tloss, metric_scores = LARS_BASE._test_loss(
                            val_dataset, model, test_metrics, device
                        )
                        log = f"epoch {epoch} | Train loss: {train_loss/total_sample:.2f}  | Test loss: {tloss:.2f}"
                        for key, val in metric_scores.items():
                            log += f"  | Test {key}: {val:.2f}"
                        print(log)
                    if wandb_run:
                        wandb_run.log(
                            {
                                "iter": iteration,
                                "train_loss": train_loss / total_sample,
                                "test_loss": tloss,
                            }.update(metric_scores)
                        )

                    if metric_scores[main_metric] > best_score:
                        best_model = copy.deepcopy(model.cpu())
                        best_score = metric_scores[main_metric]
                        if save_path:
                            best_model.save_pretrained(save_path)
                            tokenizer.save_pretrained(save_path)
                        model.to(device)
                        if save_path:
                            checkpoint = {
                                "best_metric": best_score,
                                "main_metric": main_metric,
                                "epoch": epoch,
                                "iteration": iteration,
                                "lr": lr,
                                "batch_size": batch_size,
                                "number_of_bins": number_of_bins,
                            }
                            # Save the checkpoint
                            with open(os.path.join(save_path, "training_metadata.json"), "w") as f:
                                json.dump(checkpoint, f, indent=2)

                    train_loss, total_sample = 0, 0
                    model.train()
        return best_model
    



    @staticmethod
    def train_lars_model(
        datasets: list,
        size_for_each_dataset: list,
        val_ratio: float,
        seed: int,
        chat_model_name: str,
        correctness_evaluator,
        save_path: str = None,
        wandb_run=None,
        previous_context: list = [
            {"role": "system", "content": DEFAULT_SYSTEM_BENCHMARK_PROMPT}
        ],
        user_prompt: str = DEFAULT_LLAVA_PROMPT_NO_CONTEXT,
        num_gen_per_question: int = 5,
        number_of_bins: int = 8,
        lars_model_name: str = "duygunuryldz/LARS", #"roberta-base",
        test_metrics: list[str] = ["auroc"],
        main_metric: str = "auroc",
        epochs: int = 3,
        lr: float = 5e-6,
        batch_size: int = 8,
        test_freq: int = 100,
        device="cuda",
        context_mode = None,
        dataset_csv= None,
        dataset_path = None,
        kb_path = None,
        image_directory = None,
        load_data_path = None,
        save_data_path = None,
        **kwargs,
    ):
        if load_data_path:
            train_data, val_data = LARS_BASE.load_datasets(load_data_path)
        else:
            train_data, val_data = LARS_BASE.generate_and_save_labeled_data(
            datasets=datasets,
            size_for_each_dataset=size_for_each_dataset,
            val_ratio=val_ratio,
            seed=seed,
            chat_model_name=chat_model_name,
            correctness_evaluator=correctness_evaluator,
            context_mode=context_mode,
            dataset_csv=dataset_csv,
            dataset_path=dataset_path,
            kb_path = kb_path,
            image_directory=image_directory,
            previous_context=previous_context,
            user_prompt=user_prompt,
            num_gen_per_question=num_gen_per_question,
            save_data_path=save_data_path,
            **kwargs,
            )
        # Find edges
        all_probs = []
        for d in tqdm(train_data):
            for i in range(len(d["probs"])):
                all_probs += d["probs"][i]
        edges = np.quantile(all_probs, np.linspace(0, 1, number_of_bins))
        all_probs = None

        # create LARS model
        model = AutoModelForSequenceClassification.from_pretrained(
            lars_model_name, num_labels=1
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(lars_model_name)

        
        LARS_BASE._prepare_model_and_tokenizer(
            model=model, tokenizer=tokenizer, number_of_bins=number_of_bins, edges=edges
        )

        # prepare data for LARS training, return train and val datasets

        train_dataset, val_dataset = LARS_BASE._prepare_data(
            tokenizer=tokenizer,
            train_data=train_data,
            val_data=val_data,
            number_of_bins=number_of_bins,
            edges=edges,
        )


        model = LARS_BASE._train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            model=model,
            tokenizer=tokenizer,
            save_path=save_path,
            device=device,
            test_metrics=test_metrics,
            main_metric=main_metric,
            wandb_run=wandb_run,
            number_of_bins=number_of_bins,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            test_freq=test_freq,
        )

        return model, tokenizer, train_data, val_data
