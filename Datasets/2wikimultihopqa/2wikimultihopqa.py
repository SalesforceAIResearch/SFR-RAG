import hydra
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import torch
import re
import string
from collections import Counter
from vllm import LLM, SamplingParams
import backoff
import openai
from openai import OpenAI
import os 
import argparse
import inflect
ENGINE = inflect.engine()

openai_client = None


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


def create_prompts(batch, system_prompt):
    return [
        f"{system_prompt}\nContext: {context}\nQuestion: {question}\nAnswer:"
        for question, context in zip(batch['question'], batch['context'])
    ]

@backoff.on_exception(backoff.expo, openai.RateLimitError)
def completions_with_backoff(**kwargs):
    global openai_client
    openai_client = openai_client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return openai_client.chat.completions.create(**kwargs)

def call_model_chatgpt(prompt, model, max_tokens=50):
    try:
        results = completions_with_backoff(
            model=model,
            messages=[
                {"role": "user",
                    "content": prompt},
            ],
            timeout=60,
            max_tokens=max_tokens,
        )
        result = results.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"Error: {e}")
        result = "ERROR: API error outputs"
    return result

@hydra.main(version_base=None , config_path="../../config", config_name="config.yaml")
def main(cfg):
    running_config = HydraConfig.get()
    config_name = Path(running_config.job.config_name).stem
    
    parser = argparse.ArgumentParser()
    cache_dir = cfg.paths.cache_dir
    # eval_dataset = load_dataset("xanhho/2WikiMultihopQA", split="dev", cache_dir=cache_dir)
    eval_dataset = load_dataset("ContextualBench/ContextualBench", "2WikiMultihopQA", split="validation", cache_dir=cache_dir)
    
    system_prompt = f"You are an expert in retrieval QA. Answer in fewest words possible. No context, No explanation, and No elaboration. It is crucial to provide the exact answer only."
    all_prompts = create_prompts(eval_dataset, system_prompt)
    
    pred_ans = []
    num_correct = 0
    num_total = 0
    f1_total = 0
    easy_correct = 0

    model_id = cfg.model.llm
    batch_size = cfg.batch_size

    if cfg.model.llm_type == "hf":

        llm = LLM(model=model_id, dtype=torch.bfloat16, gpu_memory_utilization=cfg.gpu_util,tensor_parallel_size=cfg.num_gpu)
        tokenizer = llm.get_tokenizer() 
        stop_tok_id = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>"), tokenizer.convert_tokens_to_ids("<|eos_token|>"),tokenizer.convert_tokens_to_ids("<end_of_turn>"),tokenizer.convert_tokens_to_ids("</s>")]

        stop_list = ["explanation", "elaboration", "context", "explanation:", "elaboration:", "context:", "question:", "answer:", "answer",
             "Explanation", "Elaboration", "Context", "Explanation:", "Elaboration:", "Context:", "Question:", "Answer:", "Answer", "confidence", "Confidence", "Confidence:", "confidence:", "<|eos_token|>", 'title', "retrieval", "eos", "<|eot_id|>"]

        sampling_params = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            stop_token_ids=stop_tok_id,
            stop = stop_list
        )

        for i in tqdm(range(0, len(all_prompts), batch_size)):
            batch_prompts = all_prompts[i:i+batch_size]
            outputs = llm.generate(batch_prompts, sampling_params)
            
            for j, output in enumerate(outputs):
                pred_answer = output.outputs[0].text.strip()
                
                pred_ans.append(pred_answer)
                pred_answer = normalize_answer(pred_answer)
                gold_answers = normalize_answer(eval_dataset[i+j]['answer'])

                if pred_answer == gold_answers:
                    num_correct += 1
                    easy_correct += 1
                    print(f"Correct! Predicted: {pred_answer}, Gold: {gold_answers}")
                elif (pred_answer in gold_answers) or (gold_answers in pred_answer):
                    easy_correct += 1
                    print(f"Partially ! Predicted: {pred_answer}, Gold: {gold_answers}")
                else:
                    print(f"Incorrect! Predicted: {pred_answer}, Gold: {gold_answers}")
                num_total += 1

                f1_this_question = f1_score(pred_answer, gold_answers)
                f1_total += f1_this_question
    elif cfg.model.llm_type == "openai":
        for i in tqdm(range(len(all_prompts))):
            prompt = all_prompts[i]
            pred_answer = call_model_chatgpt(
                        prompt, model=model_id, max_tokens=cfg.max_tokens)
            pred_ans.append(pred_answer)
            pred_answer = normalize_answer(pred_answer)
            gold_answers = normalize_answer(eval_dataset[i]['answer'])

            if pred_answer == gold_answers:
                num_correct += 1
                easy_correct += 1
                print(f"Correct! Predicted: {pred_answer}, Gold: {gold_answers}")
            elif (pred_answer in gold_answers) or (gold_answers in pred_answer):
                easy_correct += 1
                print(f"Partially ! Predicted: {pred_answer}, Gold: {gold_answers}")
            else:
                print(f"Incorrect! Predicted: {pred_answer}, Gold: {gold_answers}")
            num_total += 1

            f1_this_question = f1_score(pred_answer, gold_answers)
            f1_total += f1_this_question
    else:
        raise ValueError("Invalid LLM type")

    accuracy = num_correct / num_total
    accuracy_easy = easy_correct / num_total
    f1 = f1_total / num_total
    print(f"Model ID - {model_id} EasyM is {accuracy_easy:.3f}, Exact Match: {accuracy:.3f}, F1: {f1:.3f}")

if __name__ == "__main__":
    main()

