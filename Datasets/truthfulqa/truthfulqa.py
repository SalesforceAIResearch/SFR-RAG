from transformers import AutoModelForCausalLM, AutoTokenizer
# from deepeval import models
from deepeval.models.base_model import DeepEvalBaseLLM
from typing import List
from openai import OpenAI
import cohere
import time
from transformers import pipeline
import transformers
import torch
import hydra
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from deepeval.benchmarks import TruthfulQA
from deepeval.benchmarks.tasks import TruthfulQATask
from deepeval.benchmarks.modes import TruthfulQAMode


class LLAMA(DeepEvalBaseLLM):
    def __init__(
        self,
        cfg,
        model_id=None
    ):
        
        self.pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        tokenizer=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto")
        self.cfg = cfg
        

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:       
        model_id = self.cfg.model.llm
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

        outputs = self.pipeline(
                prompt,
                max_new_tokens=self.cfg.max_tokens,
                eos_token_id=terminators,
                do_sample=True,
                temperature=self.cfg.temperature,
                return_full_text=False,
                top_p=self.cfg.top_p,)
        
        pred_answer = outputs[0]["generated_text"].strip().lower()
        return pred_answer

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    # This is optional.
    def batch_generate(self, promtps: List[str]) -> List[str]:
        pass

    def get_model_name(self):
        return "LLAMA"
    
class GPT(DeepEvalBaseLLM):
    def __init__(
        self,
        cfg,
        model,
    ):
        self.model = model
        self.client = OpenAI(api_key=cfg.apikeys.openai)
        self.cfg = cfg

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
        model=self.model,
        messages=[
            {"role": "system", "content": "You are an expert in retrieval QA. Please respond with the exact answer only. Dont be verbose or provide extra information."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=self.cfg.max_tokens,
        n=1,
        temperature=self.cfg.temperature,
        )
        pred_answer = response.choices[0].message.content.strip()    
        return pred_answer

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    # This is optional.
    def batch_generate(self, promtps: List[str]) -> List[str]:
        pass

    def get_model_name(self):
        return "GPT"
    
class Cohere(DeepEvalBaseLLM):
    def __init__(
        self,
        cfg,
        model,
    ):
        self.model = model
        self.co = cohere.Client(cfg.apikeys.cohere)
        self.cfg = cfg

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        try:

            response = self.co.chat(
                model=self.model,
                message=f"You are an expert in retrieval QA. Please respond with the exact answer only. Don't be verbose or provide extra information.\n {prompt}"
            )
            print(f"response: {response.text.strip()}")
            pred_answer = response.text.strip()
        except Exception as e:
            print(f"Error: {e}")
            pred_answer = "Error"
        return pred_answer

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    # This is optional.
    def batch_generate(self, promtps: List[str]) -> List[str]:
        pass

    def get_model_name(self):
        return "Cohere"



@hydra.main(version_base=None , config_path="../../config", config_name="config.yaml")
def main(
    cfg
):
    hydra.output_subdir = "null"
    benchmark = TruthfulQA(
        mode=TruthfulQAMode.MC1
    )
    
    model_id = cfg.model.llm
    model_type = cfg.model.llm_type
    
    if model_type == "openai":
        model = GPT(cfg,model=model_id)
    elif model_type == "cohere":
        model = Cohere(cfg,model=model_id)
    elif model_type == "hf":
        model = LLAMA(cfg,model_id)
    else:
        print("Invalid model type")
        return
    
    benchmark.evaluate(model=model)

    print(benchmark.overall_score)
    print(f"Model ID - {model_id}")

if __name__ == "__main__":
    main()