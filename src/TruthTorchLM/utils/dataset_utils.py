from datasets import load_dataset
from TruthTorchLM.availability import AVAILABLE_DATASETS, AVAILABLE_CONTEXTS_EVQA, AVAILABLE_CONTEXTS_INFOSEEK
import pandas as pd
from typing import Union
from tqdm import tqdm
from PIL import Image
import os
import json
import ast
import pickle



MAX_CONTEXT_LENGTH = 2000


def _require_path(value, dataset_name: str, argument_name: str):
    if value is None:
        raise ValueError(
            f"{argument_name} must be provided when loading the {dataset_name} dataset."
        )
    return value


def get_dataset(
    dataset: Union[str, list], size_of_data: float = 1.0, seed: int = 0, split="test", context_mode = None, dataset_path = None, image_directory = None, dataset_csv = None, retrieval_file = None, kb_path = None
):
    if type(dataset) != str:
        if len(dataset) == 0:
            raise ValueError("Dataset list is empty.")
        if (
            "question" not in dataset[0].keys()
            or "ground_truths" not in dataset[0].keys()
        ):
            raise ValueError(
                "Dataset should have 'question' and 'ground_truths' keys.")
        return dataset

    if dataset not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Dataset is not available. Available datasets are: {AVAILABLE_DATASETS}"
        )
    
    if dataset not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Dataset is not available. Available datasets are: {AVAILABLE_DATASETS}"
        )
    

    print(
        "Loading dataset, split:",
        split,
        "fraction of data:",
        size_of_data,
    )

    if dataset == "trivia_qa":
        dataset = get_trivia_qa(
            size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "gsm8k":
        dataset = get_gsm8k(size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "natural_qa":
        dataset = get_natural_qa(
            size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "pop_qa":
        dataset = get_pop_qa(size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "simple_qa":
        dataset = get_simple_qa(
            size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "wikipedia_factual":
        dataset = get_wikipedia_factual(
            size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "narrative_qa":
        dataset = get_narrative_qa(
            size_of_data=size_of_data, seed=seed, split=split)
    elif dataset == "okvqa":
        dataset = get_okvqa(
            size_of_data=size_of_data, seed=seed)
    elif dataset == "evqa":
        dataset = get_evqa(
            size_of_data=size_of_data, seed=seed, dataset_path = dataset_path, image_directory=image_directory, context_mode=context_mode, dataset_csv=dataset_csv, retrieval_file = retrieval_file)
    elif dataset == "infoseek":
        dataset = get_infoseek(
            size_of_data=size_of_data, seed=seed, dataset_path = dataset_path, image_directory=image_directory, context_mode=context_mode, dataset_csv=dataset_csv, retrieval_file = retrieval_file, kb_path=kb_path)

    return dataset


def get_trivia_qa(size_of_data: float = 1.0, seed: int = 0, split="test"):

    if split == "test":
        raw_dataset = load_dataset(
            "trivia_qa", "rc.nocontext", split="validation")
    elif split == "train":
        raw_dataset = load_dataset("trivia_qa", "rc.nocontext", split="train")
    else:
        raise ValueError("Split should be either 'test' or 'train'.")

    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)[
            "train"
        ]
    dataset = []
    answers = raw_dataset["answer"]
    questions = raw_dataset["question"]
    for i in tqdm(range(len(raw_dataset))):
        ground_truths = answers[i]["aliases"]
        dataset.append(
            {"context": "", "question": questions[i], "ground_truths": ground_truths})

    return dataset


def get_gsm8k(size_of_data: float = 1.0, seed: int = 0, split="test"):
    if split == "test":
        raw_dataset = load_dataset("openai/gsm8k", "main", split="test")
    elif split == "train":
        raw_dataset = load_dataset("openai/gsm8k", "main", split="train")
    else:
        raise ValueError("Split should be either 'test' or 'train'.")
    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)[
            "train"
        ]
    dataset = []
    answers = raw_dataset["answer"]
    questions = raw_dataset["question"]
    for i in tqdm(range(len(raw_dataset))):
        answer = answers[i].split("####")[1].strip()
        dataset.append({"context": "", "question": questions[i], "ground_truths": [answer]})

    return dataset


def get_natural_qa(size_of_data: float = 1.0, seed: int = 0, split="test"):
    if split == "test":
        raw_dataset = load_dataset(
            "google-research-datasets/nq_open", split="validation"
        )
    elif split == "train":
        raw_dataset = load_dataset(
            "google-research-datasets/nq_open", split="train")
    else:
        raise ValueError("Split should be either 'test' or 'train'.")
    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)[
            "train"
        ]
    dataset = []
    questions = raw_dataset["question"]
    answers = raw_dataset["answer"]
    for i in tqdm(range(len(raw_dataset))):
        dataset.append({"context": "", "question": questions[i], "ground_truths": answers[i]})

    return dataset


def get_pop_qa(size_of_data: float = 1.0, seed: int = 0, split="test"):
    if split == "test":
        raw_dataset = load_dataset("akariasai/PopQA", split="test")
    elif split == "train":
        raw_dataset = load_dataset("akariasai/PopQA", split="test")
        print("Train split is not available for PopQA. Using test split instead.")
    else:
        raise ValueError("Split should be either 'test' or 'train'.")
    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)[
            "train"
        ]
    dataset = []
    questions = raw_dataset["question"]
    answers = raw_dataset["possible_answers"]
    for i in tqdm(range(len(raw_dataset))):
        dataset.append(
            {"context": "", "question": questions[i], "ground_truths": [answers[i]]})

    return dataset


def get_simple_qa(size_of_data: float = 1.0, seed: int = 0, split="test"):
    if split == "test":
        raw_dataset = load_dataset("basicv8vc/SimpleQA", split="test")
    elif split == "train":
        raw_dataset = load_dataset("basicv8vc/SimpleQA", split="test")
        print("Train split is not available for PopQA. Using test split instead.")
    else:
        raise ValueError("Split should be either 'test' or 'train'.")
    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)[
            "train"
        ]
    dataset = []
    questions = raw_dataset["problem"]
    answers = raw_dataset["answer"]
    for i in tqdm(range(len(raw_dataset))):
        dataset.append(
            {"context": "", "question": questions[i], "ground_truths": [answers[i]]})

    return dataset


def get_wikipedia_factual(size_of_data: float = 1.0, seed: int = 0, split='train'):
    raw_dataset = load_dataset("achorn123/wikipedia_factual_dataset_500", split='train')

    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)['train']

    dataset = []
    for data in tqdm(raw_dataset, desc="Processing Wikipedia factual 500"):
        context = data["context"].strip()
        question = data["question"].strip()
        answer = data["answer"].strip()
        dataset.append({
            'context': context,
            'question': question,
            'ground_truths': [answer]
        })

    return dataset


def get_narrative_qa(size_of_data: float = 1.0, seed: int = 0, split='test'):
    raw_dataset = load_dataset("deepmind/narrativeqa", split=split)

    if size_of_data != 1.0 or type(size_of_data) != float:
        raw_dataset = raw_dataset.train_test_split(train_size=size_of_data, seed=seed)['train']

    dataset = []
    for data in tqdm(raw_dataset, desc="Processing NarrativeQA"):
        context = data["document"]["text"].strip()
        question = data["question"]["text"].strip()
        answers = []
        for answer in data["answers"]:
            answers.append(answer["text"].strip())

        dataset.append({
            "context": context,
            "question": question,
            "ground_truths": answers
        })
        
    return dataset


def get_okvqa(size_of_data: float = 1.0, seed: int = 0):

    raw_dataset = load_dataset("lmms-lab/OK-VQA")['val2014'] #only val available (5046 samples)

    if isinstance(size_of_data, range):
        raw_dataset = raw_dataset.select(size_of_data)
        size_of_data = len(size_of_data)

    if size_of_data == 1.0:
        size_of_data = len(raw_dataset)    

    dataset = []
    for i in tqdm(range(size_of_data), desc="Processing OKVQA"):
        question = raw_dataset[i]["question"].strip()
        image = raw_dataset[i]["image"]
        answers = []
        for answer in raw_dataset[i]["answers"]:
            answers.append(answer.strip())

        dataset.append({
            "image": image,
            "context": "",
            "question": question,
            "ground_truths": answers
        })
    #validation save used for mushtaq
    #with open("okvqa.pkl", "wb") as f:
    #    pickle.dump(dataset, f)
    return dataset



def get_evqa(size_of_data=500, seed: int = 0, dataset_path=None, image_directory = None, context_mode=None, dataset_csv = None, retrieval_file = None):
    if context_mode not in AVAILABLE_CONTEXTS_EVQA:
        raise ValueError(
            f"Context mode is not available. Available context modes are: {AVAILABLE_CONTEXTS_EVQA}"
        )
    else:
        print(f"Using context mode: {context_mode}")
    dataset_path = _require_path(dataset_path, "EVQA", "dataset_path")
    image_directory = _require_path(image_directory, "EVQA", "image_directory")
    
    # Load the main dataset
    if dataset_csv is None:
        dataset_csv="evqa_small.csv"
    df = pd.read_csv(os.path.join(dataset_path, dataset_csv))
    mapping_df = pd.read_csv(os.path.join(dataset_path, "evqa_kb_sections.csv"))  

    # Handle size_of_data input
    if isinstance(size_of_data, range):
        start, stop = size_of_data.start, size_of_data.stop
        df = df.iloc[start:stop]
    elif isinstance(size_of_data, float):
        subset_size = int(len(df) * size_of_data)
        df = df.iloc[:subset_size]
    elif isinstance(size_of_data, int):
        df = df.iloc[:size_of_data]

    dataset = []

    # Base image folder
    image_base_path = os.path.join(image_directory, "e_vqa_images")

    # ✅ Properly decode JSON lists from the KB CSV
    url_to_sections = {}
    for _, row in mapping_df.iterrows():
        section_titles = ast.literal_eval(row['section_titles'])
        section_texts = ast.literal_eval(row['section_texts'])
        url_to_sections[row['wikipedia_url']] = (section_titles, section_texts)


    if context_mode == "random_doc":
        if retrieval_file:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/" , retrieval_file)
        else:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/evqa_random_sections.csv")
        retrieval_sections_df = pd.read_csv(retrieval_csv_path)

    if context_mode == "doc-":
        if retrieval_file:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/" , retrieval_file)
        else:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/bm25_hard_negatives.csv")
        retrieval_sections_df = pd.read_csv(retrieval_csv_path)

    if context_mode == "bm25":
        if retrieval_file:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/" , retrieval_file)
        else:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/bm25_docs_large_without_caption.csv")
        retrieval_sections_df = pd.read_csv(retrieval_csv_path)
    
    if context_mode == "rerank":
        if retrieval_file:
            retrieval_csv_path =os.path.join(dataset_path, "retrieval_results/" , retrieval_file)
        else:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/rerank_docs_without_caption.csv")
        retrieval_sections_df = pd.read_csv(retrieval_csv_path)

    if context_mode == "mixed":
        retrieval_csv_path =os.path.join(dataset_path, "retrieval_results/" , retrieval_file) #no base retrieval file for mixed
        retrieval_sections_df = pd.read_csv(retrieval_csv_path)

    if context_mode == "eva_clip+contriever":
        if retrieval_file:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/" , retrieval_file)
        else:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/evqa_eva_clip_contriever.csv")
        retrieval_sections_df = pd.read_csv(retrieval_csv_path)            

    for idx, row in df.iterrows():
        image_path = row['selected_image_path']
        question = row['question']
        answer = row['answer']
        wiki_url = row['wikipedia_url']
        evidence_section_title = row['evidence_section_title']

        context = ""
        if context_mode == "doc+":
            section_titles, section_texts = url_to_sections[wiki_url]
            idx = section_titles.index(evidence_section_title)
            context = section_texts[idx]

        elif context_mode == "random_doc":
            retrieval_row = retrieval_sections_df.iloc[idx]
            retrieval_url = retrieval_row["wikipedia_url"]
            section_title = retrieval_row["section_title"]
            section_titles, section_texts = url_to_sections[retrieval_url]
            retrieval_idx = section_titles.index(section_title)
            context = section_texts[retrieval_idx]

        elif context_mode == "bm25":
            retrieval_row = retrieval_sections_df.iloc[idx]
            retrieval_url = ast.literal_eval(retrieval_row["wikipedia_url"])[0]
            section_title = ast.literal_eval(retrieval_row["section_title"])[0]
            bm25_score = 0 
            section_titles, section_texts = url_to_sections[retrieval_url]
            retrieval_idx = section_titles.index(section_title)
            context = section_texts[retrieval_idx]

        elif context_mode == "rerank":
            retrieval_row = retrieval_sections_df.iloc[idx]
            retrieval_url = ast.literal_eval(retrieval_row["wikipedia_url"])[0]
            section_title = ast.literal_eval(retrieval_row["section_title"])[0]
            section_titles, section_texts = url_to_sections[retrieval_url]
            retrieval_idx = section_titles.index(section_title)
            context = section_texts[retrieval_idx] 
            
        elif context_mode == "eva_clip+contriever":          
            retrieval_row = retrieval_sections_df.iloc[idx]
            retrieval_url = retrieval_row['article_url']
            section_title = retrieval_row['section_title']
            section_titles, section_texts = url_to_sections[retrieval_url]
            retrieval_idx = section_titles.index(section_title)
            context = section_texts[retrieval_idx] 
            perturb_score = 0 

        elif context_mode == "doc-":
            retrieval_row = retrieval_sections_df.iloc[idx]
            retrieval_url = ast.literal_eval(retrieval_row["wikipedia_url"])[0]
            section_title = ast.literal_eval(retrieval_row["section_title"])[0]
            section_titles, section_texts = url_to_sections[retrieval_url]
            retrieval_idx = section_titles.index(section_title)
            context = section_texts[retrieval_idx]

        if len(context) > MAX_CONTEXT_LENGTH:
            print("Context too long")
            if answer in context:
                answer_start = context.find(answer)
                answer_end = answer_start + len(answer)
                
                half_len = MAX_CONTEXT_LENGTH // 2
                start = max(0, answer_start - half_len)
                end = min(len(context), answer_end + half_len)
                
                if end - start < MAX_CONTEXT_LENGTH:
                    if start == 0:
                        end = min(len(context), start + MAX_CONTEXT_LENGTH)
                    elif end == len(context):
                        start = max(0, end - MAX_CONTEXT_LENGTH)
                
                context = context[start:end]
            else:
                start = 0
                context = context[start:start + MAX_CONTEXT_LENGTH]
                print("Answer not found in context, using random substring")

        # Build the full path to the image
        full_image_path = os.path.join(image_base_path, image_path)

        # Load the image safely
        image = Image.open(full_image_path).convert("RGB")
        entry = {
            "image": image,
            "context": context,
            "question": question,
            "ground_truths": [answer]
        }
        if context_mode == "bm25":
            entry["bm25_score"] = bm25_score
        if context_mode == "eva_clip+contriever":
            entry["perturb_score"] = perturb_score
        dataset.append(entry)
    return dataset



def get_infoseek(size_of_data=500, seed: int = 0, dataset_path=None,  image_directory = None, context_mode=None, dataset_csv = None, retrieval_file = None, kb_path = None):
    if context_mode not in AVAILABLE_CONTEXTS_EVQA:
        raise ValueError(
            f"Context mode is not available. Available context modes are: {AVAILABLE_CONTEXTS_EVQA}"
        )
    else:
        print(f"Using context mode: {context_mode}")
    dataset_path = _require_path(dataset_path, "InfoSeek", "dataset_path")
    image_directory = _require_path(image_directory, "InfoSeek", "image_directory")
    kb_path = _require_path(kb_path, "InfoSeek", "kb_path")
    
    # Load the main dataset
    if dataset_csv is None:
        dataset_csv="infoseek_val_evidence.csv"
    df = pd.read_csv(os.path.join(dataset_path, dataset_csv))

    mapping_df = pd.read_csv(os.path.join(kb_path, "evqa_kb_sections.csv")) #for our setup we use the same kb for both datasets

    # Handle size_of_data input
    if isinstance(size_of_data, range):
        start, stop = size_of_data.start, size_of_data.stop
        df = df.iloc[start:stop]
    elif isinstance(size_of_data, float):
        subset_size = int(len(df) * size_of_data)
        df = df.iloc[:subset_size]
    elif isinstance(size_of_data, int):
        df = df.iloc[:size_of_data]

    dataset = []

    # Base image folder
    image_base_path = image_directory


    # ✅ Properly decode JSON lists from the KB CSV
    url_to_sections = {}
    for _, row in mapping_df.iterrows():
        section_titles = ast.literal_eval(row['section_titles'])
        section_texts = ast.literal_eval(row['section_texts'])
        url_to_sections[row['wikipedia_url']] = (section_titles, section_texts)


    if context_mode == "random_doc":
        if retrieval_file:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/" , retrieval_file)
        else:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/evqa_random_sections.csv")
        retrieval_sections_df = pd.read_csv(retrieval_csv_path)

    if context_mode == "doc-":
        if retrieval_file:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/" , retrieval_file)
        else:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/bm25_hard_negatives.csv")
        retrieval_sections_df = pd.read_csv(retrieval_csv_path)

    if context_mode == "bm25":
        if retrieval_file:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/" , retrieval_file)
        else:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/bm25_docs_large_without_caption.csv")
        retrieval_sections_df = pd.read_csv(retrieval_csv_path)
    
    if context_mode == "rerank":
        if retrieval_file:
            retrieval_csv_path =os.path.join(dataset_path, "retrieval_results/" , retrieval_file)
        else:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/rerank_docs_without_caption.csv")
        retrieval_sections_df = pd.read_csv(retrieval_csv_path)

    if context_mode == "mixed":
        retrieval_csv_path =os.path.join(dataset_path, "retrieval_results/" , retrieval_file) #no base retrieval file for mixed
        retrieval_sections_df = pd.read_csv(retrieval_csv_path)

    if context_mode == "eva_clip+contriever":
        if retrieval_file:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/" , retrieval_file)
        else:
            retrieval_csv_path = os.path.join(dataset_path, "retrieval_results/evqa_eva_clip_contriever.csv")
        retrieval_sections_df = pd.read_csv(retrieval_csv_path)            

    for idx, row in df.iterrows():
        image_path = row['image_path']
        question = row['question']
        answer = row['answer']
        wiki_url = row['wikipedia_url']
        evidence_section_title = row['evidence_section_title']

        context = ""
        if context_mode == "doc+":
            section_titles, section_texts = url_to_sections[wiki_url]
            idx = section_titles.index(evidence_section_title)
            context = section_texts[idx]
        elif context_mode == "random_doc":
            retrieval_row = retrieval_sections_df.iloc[idx]
            retrieval_url = retrieval_row["wikipedia_url"]
            section_title = retrieval_row["section_title"]
            section_titles, section_texts = url_to_sections[retrieval_url]
            retrieval_idx = section_titles.index(section_title)
            context = section_texts[retrieval_idx]

        elif context_mode == "bm25":
            retrieval_row = retrieval_sections_df.iloc[idx]
            retrieval_url = ast.literal_eval(retrieval_row["wikipedia_url"])[0]
            section_title = ast.literal_eval(retrieval_row["section_title"])[0]
            bm25_score = 0 
            section_titles, section_texts = url_to_sections[retrieval_url]
            retrieval_idx = section_titles.index(section_title)
            context = section_texts[retrieval_idx]

        elif context_mode == "rerank":
            retrieval_row = retrieval_sections_df.iloc[idx]
            retrieval_url = ast.literal_eval(retrieval_row["wikipedia_url"])[0]
            section_title = ast.literal_eval(retrieval_row["section_title"])[0]
            section_titles, section_texts = url_to_sections[retrieval_url]
            retrieval_idx = section_titles.index(section_title)
            context = section_texts[retrieval_idx] 
            
        elif context_mode == "eva_clip+contriever":          
            retrieval_row = retrieval_sections_df.iloc[idx]
            retrieval_url = retrieval_row['article_url']
            section_title = retrieval_row['section_title']
            section_titles, section_texts = url_to_sections[retrieval_url]
            retrieval_idx = section_titles.index(section_title)
            context = section_texts[retrieval_idx] 
            perturb_score = 0 

        elif context_mode == "doc-":
            retrieval_row = retrieval_sections_df.iloc[idx]
            retrieval_url = ast.literal_eval(retrieval_row["wikipedia_url"])[0]
            section_title = ast.literal_eval(retrieval_row["section_title"])[0]
            section_titles, section_texts = url_to_sections[retrieval_url]
            retrieval_idx = section_titles.index(section_title)
            context = section_texts[retrieval_idx]

        if len(context) > MAX_CONTEXT_LENGTH:
            print("Context too long")
            if answer in context:
                answer_start = context.find(answer)
                answer_end = answer_start + len(answer)
                
                half_len = MAX_CONTEXT_LENGTH // 2
                start = max(0, answer_start - half_len)
                end = min(len(context), answer_end + half_len)
                
                if end - start < MAX_CONTEXT_LENGTH:
                    if start == 0:
                        end = min(len(context), start + MAX_CONTEXT_LENGTH)
                    elif end == len(context):
                        start = max(0, end - MAX_CONTEXT_LENGTH)
                
                context = context[start:end]
            else:
                start = 0
                context = context[start:start + MAX_CONTEXT_LENGTH]
                print("Answer not found in context, using random substring")

        # Build the full path to the image
        full_image_path = os.path.join(image_base_path, image_path)

        # Load the image safely
        image = Image.open(full_image_path).convert("RGB")
        entry = {
            "image": image,
            "context": context,
            "question": question,
            "ground_truths": [answer]
        }
        if context_mode == "bm25":
            entry["bm25_score"] = bm25_score
        if context_mode == "eva_clip+contriever":
            entry["perturb_score"] = perturb_score
        dataset.append(entry)
    return dataset
