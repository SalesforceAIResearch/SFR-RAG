from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
import cohere
import time
from openai import OpenAI
from transformers import pipeline
import accelerate
import os
import sys
import transformers
import torch
import re
import string
from collections import Counter
import inflect
ENGINE = inflect.engine()

def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gt_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def normalize_answer(s,p=ENGINE):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')
    
    def convert_numbers_to_words(text):
        words = text.split()
        result = []
        for word in words:
            if word.isdigit() and int(word) < 100:
                word_in_words = p.number_to_words(int(word))
                result.append(word_in_words)
            else:
                result.append(word)
        return ' '.join(result)

    return white_space_fix(remove_articles(handle_punc(convert_numbers_to_words(lower(replace_underscore(s)))))).strip()


@hydra.main(version_base=None , config_path="../../config", config_name="config.yaml")
def main(
    cfg
):
    running_config = HydraConfig.get()
    config_name = Path(running_config.job.config_name).stem    
    model_name = cfg.model.llm
    cache_dir = cfg.paths.cache_dir
    model_id = cfg.model.llm
    
    print(f"Running model {model_name}")
    
    eval_dataset = load_dataset("Salesforce/ContextualBench", "hotpotqa" ,split="validation", cache_dir=cache_dir)
    
    print(eval_dataset)
    pred_ans = []
    num_correct = 0
    num_total = 0
    f1_total = 0
    easy_correct = 0
    
    if cfg.model.llm_type == "hf":
        pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto")
    
    for idx,example in tqdm(enumerate(eval_dataset)):
        question = example["question"]
        context = example["retrieved_passages"]
        prompt = f"You are an expert in retrieval QA. Please respond with the exact answer only. Dont be verbose or provide extra information. \n Context: {context}\nQuestion: {question}\nAnswer:"
        
        if cfg.model.llm_type == "openai":
            client = OpenAI(api_key=cfg.apikeys.openai)
            response = client.chat.completions.create(
            model=cfg.model.llm,
            messages=[
                {"role": "system", "content": "You are an expert in retrieval QA. Please respond with the exact answer only. Dont be verbose or provide extra information."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=cfg.max_new_tokens,
            n=1,
            temperature=cfg.temperature,
            )
            pred_answer = response.choices[0].message.content.strip()
        elif cfg.model.llm_type == "cohere":
            while True:
                try:
                    key= cfg.apikeys.cohere
                    co = cohere.Client(key)
                    response = co.chat(
                        model=cfg.model.llm,
                        message=f"You are an expert in retrieval QA. Please respond with the exact answer only. Don't be verbose or provide extra information.\n {prompt}"
                    )
                    pred_answer = response.text.strip()
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(6) # Sleep for 6 seconds to avoid rate limiting
                    continue
                break
            
        elif cfg.model.llm_type == "hf":
            messages = [
            {"role": "system", "content": "You are an expert in retrieval QA.Please respond with the exact answer only. Dont be verbose or provide extra information."},
            {"role": "user", "content": f"{prompt}"},]

            if "gemma" in model_id.lower():
                terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
                ]
            elif "llama" in model_id.lower():
                terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]
            elif "mistral" in model_id.lower():
                terminators = [
                pipeline.tokenizer.eos_token_id,
                pipeline.tokenizer.convert_tokens_to_ids("</s>"),
                ]
            else:
                terminators = [
                pipeline.tokenizer.eos_token_id,]

            outputs = pipeline(
            messages,
            max_new_tokens=cfg.max_new_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,)

            pred_answer = outputs[0]["generated_text"][-1]['content'].strip()
        else:
            raise ValueError("Invalid LLM type")

        pred_ans.append(pred_answer)
        pred_answer = normalize_answer(pred_answer)
        gold_answers = normalize_answer(example["answer"])

        if pred_answer == gold_answers:
            num_correct += 1
            easy_correct += 1
            print(f"Correct! Predicted: {pred_answer}, Gold: {gold_answers}")
        elif (pred_answer in gold_answers) or (gold_answers in pred_answer):
            print(f"Partial Correct! Predicted: {pred_answer}, Gold: {gold_answers}")
            easy_correct += 1
        else:
            print(f"Incorrect! Predicted: {pred_answer}, Gold: {gold_answers}")
        num_total += 1

        f1_this_question = f1_score(pred_answer, gold_answers)
        f1_total += f1_this_question

    accuracy = num_correct / num_total
    accuracy_easy = easy_correct / num_total
    f1 = f1_total / num_total
    print(f"Model ID - {model_id} EasyM is {accuracy_easy:.3f} ,the Exact Match: {accuracy:.3f}, F1: {f1:.3f}")
    
if __name__ == "__main__":
    main()
