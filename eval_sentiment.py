from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score, classification_report, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from constants import INFER_PROMPT_TEMPLATE, SPLIT_ON_TERM
import os


MAX_SEQ_LENGTH = 512  # Choose any! We auto support RoPE Scaling internally!
dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
# Use 4bit quantization to reduce memory usage. Can be False.
load_in_4bit = True


def load_model(model_path: str, infer_mode: bool = True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )
    if infer_mode:
        FastLanguageModel.for_inference(model)
    return model, tokenizer


def get_data(data_file_path: str) -> pd.DataFrame:
    def convert_to_lowercase(example):
        example["LABEL"] = example["LABEL"].lower()
        return example
    data = load_dataset('json', data_files=data_file_path)
    data = data.filter(lambda example: example["LABEL"] is not None)
    data = data.map(convert_to_lowercase)
    data = pd.DataFrame(data["train"])
    return data


def eval_model(data: pd.DataFrame, tokenizer, model, prompt_template: str, split_term: str):
    num_failed_classifications = 0
    for index, row in tqdm(data.iterrows(), total=data.shape[0], desc="Making Predictions"):
        inputs = tokenizer(
            [prompt_template.format(text=row['INDIC REVIEW'])],
            return_tensors="pt"
        ).to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=5,  # We only need a short response
            num_return_sequences=1,
            do_sample=False,  # Use greedy decoding
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = decoded_output.split(
            split_term)[-1].strip()
        classification = response.lower().strip()
        if "positive" in classification and "negative" not in classification:
            classification = "positive"
        elif "negative" in classification and "positive" not in classification:
            classification = "negative"
        else:
            print(response)
            classification = None
            num_failed_classifications += 1
            print(f"Classification Failed for Index: {index}")
        data.at[index, 'pred'] = classification
    print(f"Number of Failed Classification: {num_failed_classifications}")
    return data


def eval_scores(data: pd.DataFrame, average_method: str = "macro"):
    labels = list(data["LABEL"])
    preds = list(data["pred"])
    labels = list(map(lambda x: 1 if x == "positive" else 0, labels))
    preds = list(map(lambda x: 1 if x == "positive" else 0, preds))

    f1 = f1_score(labels, preds, average=average_method)
    recall = recall_score(labels, preds, average=average_method)
    precision = precision_score(labels, preds, average=average_method)
    print(f"Average: {average_method}")
    print(f"Precision: {precision}\nRecall: {recall}\nF1Score: {f1}\n\n")
    return precision, recall, f1


def get_accuracy(data):
    labels = list(data["LABEL"])
    preds = list(data["pred"])
    labels = list(map(lambda x: 1 if x == "positive" else 0, labels))
    preds = list(map(lambda x: 1 if x == "positive" else 0, preds))
    acc = accuracy_score(labels, preds)
    print(f"Accuracy: {acc}\n")
    return acc


def save_cnf_matrix(data, save_path: str = None):
    labels = list(data["LABEL"])
    preds = list(data["pred"])
    labels = list(map(lambda x: 1 if x == "positive" else 0, labels))
    preds = list(map(lambda x: 1 if x == "positive" else 0, preds))
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=data["LABEL"].unique(),
                yticklabels=data["pred"].unique())
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
        return
    plt.savefig("cmf_matrix_latest.png")


def infer(model_path: str, data_path: str):
    model, tokenizer = load_model(model_path=model_path)
    data = get_data(data_path)
    data = eval_model(data=data, tokenizer=tokenizer, model=model,
                      prompt_template=INFER_PROMPT_TEMPLATE, split_term=SPLIT_ON_TERM)

    get_accuracy(data)
    eval_scores(data, average_method="macro")
    eval_scores(data, average_method="weighted")
    eval_scores(data, average_method="micro")

    os.makedirs("outputs", exist_ok=True)
    data.to_csv(
        f"outputs/{model_path.split('/')[-1].replace('-', '_')}_preds.csv")
    save_cnf_matrix(
        data, save_path=f"outputs/{model_path.split('/')[-1].replace('-', '_')}.png")
    save_cnf_matrix(
        data, save_path=f"{model_path.split('/')[-1].replace('-', '_')}.png")
