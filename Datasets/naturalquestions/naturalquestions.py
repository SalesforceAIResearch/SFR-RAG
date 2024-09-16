import openai
import sys
from datasets import load_dataset, Dataset
from collections import Counter
from openai import OpenAI
import string
import re
import cohere
import time
from tqdm import tqdm
import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import transformers
import torch
import inflect
ENGINE = inflect.engine()

def get_predictions(cfg,dataset,pipeline=None):
    example_task = """ 
    Question: when is the last episode of season 8 of the walking dead?
    Answer: March 18, 2018

    Question: 100 acres is equal to how many hectares?
    Answer: 1 square hectometre (hm2)

    Question: where was donovan mitchell picked in the draft?
    Answer: 13th

    Question: what type of hybrid is the ford fusion?
    Answer: gasoline-electric

    Question: how many states is there in united states?
    Answer: 50

    """
    if cfg.model.llm_type == "openai":
        predictions = {}
        try:
            for question in tqdm(dataset):
                qid = question['id']
                client = OpenAI(api_key=cfg.apikeys.openai)
                prompt = f"""Based on the given context answer the question, Context {" ".join(question['retrieved_passages'][:cfg.retriever_top])}  Question {question['question']} Answer:"""
                response = client.chat.completions.create(
                    model=cfg.model.llm,
                    messages=[
                        {"role": "system", "content": "You are an expert in retrieval QA. Please respond with the exact answer only. Dont be verbose or provide extra information."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=cfg.max_tokens,
                    n=1,
                    temperature=cfg.temperature,
                    )
                
                answer = response.choices[0].message.content.strip()
                predictions[qid] = answer.lower()
        except Exception as e:
            print(f"Error: {e}")
            predictions[qid] = "na"
        return predictions
    elif cfg.model.llm_type == "cohere":
        predictions = {}
        try:
            for idx,question in tqdm(enumerate(dataset)):
                key = cfg.apikeys.cohere
                qid = question['id']
                co = cohere.Client(key)
                prompt = "You are an expert in retrieval QA. Please respond with the exact answer only. Don't be verbose or provide extra information.\n Given below are some examples of the task \n" + example_task + "\n" + "Now solve this question\n"
                prompt += f""" Based on the given context answer the question, Context {" ".join(question['retrieved_passages'][:cfg.retriever_top])} Question {question['question']} Answer:"""
                response = co.chat(
                        model=cfg.model.llm,
                        message=prompt,
                        max_tokens=cfg.max_tokens,
                        temperature=cfg.temperature,
                    )
                answer = response.text.strip()
                predictions[qid] = answer.lower()
                
        except Exception as e:
            print(f"Error: {e}")
            predictions[qid] = "na"
        return predictions
    elif cfg.model.llm_type == "hf":
        model_id = cfg.model.llm
        predictions = {}
        for question in tqdm(dataset):
            qid = question['id']
            prompt = "You are an expert in retrieval QA. Please respond with the exact answer only. Don't be verbose or provide extra information.\n Given below are some examples of the task \n" + example_task + "\n" + "Now solve this question\n"
            prompt += f""" Based on the given context answer the question, Context {" ".join(question['retrieved_passages'][:cfg.retriever_top])} Question {question['question']} Answer:"""
            
            messages = [
            {"role": "system", "content": "You are an expert in retrieval QA. Please respond with the exact answer only. Dont be verbose or provide extra information."},
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
            max_new_tokens=cfg.max_tokens,
            eos_token_id=terminators,
            do_sample=True,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            )

            answer = outputs[0]["generated_text"][-1]['content'].strip().lower()

            predictions[qid] = answer
            
        return predictions
    else:
        raise ValueError("Invalid LLM type")
    

def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

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

def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        prediction = normalize_answer(prediction)
        ground_truth = normalize_answer(ground_truth)

        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)

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


def get_ground_truths(answer):
    return [normalize_answer(answer)] + answer

def evaluate_nq(cfg,ground_truth, predicted_answers, qid_list=None, mute=False):
    f1 = exact_match = common = 0
    if qid_list is None:
        qid_list = ground_truth.keys()
    for qid in qid_list:
        if qid not in predicted_answers:
            if not mute:
                message = f'Missed question {qid} will receive score 0.'
                # print(message, file=sys.stderr)
            continue
        if qid not in ground_truth:
            if not mute:
                message = f'Irrelevant question {qid} will receive score 0.'
                print(message, file=sys.stderr)
            continue
        common += 1
        prediction = predicted_answers[qid]
        ground_truths = ground_truth[qid]['short_answers']
        em_for_this_question = metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        if em_for_this_question == 0 and not mute:
            print(f"em=0: {prediction} {ground_truths}")
        exact_match += em_for_this_question
        f1_for_this_question = metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)
        f1 += f1_for_this_question

    exact_match = 100.0 * exact_match / len(qid_list)
    f1 = 100.0 * f1 / len(qid_list)

    return {'exact_match': exact_match, 'f1': f1, 'common': common, 'denominator': len(qid_list),
            'pred_len': len(predicted_answers), 'gold_len': len(ground_truth)}

@hydra.main(version_base=None , config_path="../../config", config_name="config.yaml")
def main(
    cfg
):
    running_config = HydraConfig.get()
    config_name = Path(running_config.job.config_name).stem
    model_name = cfg.model.llm
    cache_dir = cfg.paths.cache_dir

    dataset = load_dataset("Salesforce/ContextualBench", "NQ" ,split="validation", cache_dir=cache_dir)
    print(dataset)

    ground_truth = {item['id']: item for item in dataset}

    sample_questions = {qid: item for qid, item in list(ground_truth.items())} 

    model_id = cfg.model.llm

    if cfg.model.llm_type == "hf":
        pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto")

        predicted_answers = get_predictions(cfg,dataset,pipeline)
    else:
        model_id = cfg.model.llm
        predicted_answers = get_predictions(cfg,dataset)

    results = evaluate_nq(cfg,ground_truth, predicted_answers, qid_list=sample_questions.keys())
    print(results)
    print(f"Model id : {model_id}")
    
if __name__ == "__main__":
    main()