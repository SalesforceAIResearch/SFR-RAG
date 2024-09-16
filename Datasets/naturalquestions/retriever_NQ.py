from datasets import load_dataset, Dataset
from transformers import TrainingArguments, Trainer
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.cohere import CohereEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import default_data_collator
import os
from datasets import Dataset, DatasetDict
from tqdm import tqdm
import cohere
import evaluate
import time
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import requests
import json 



# Import the keys and setting the environment variables

# Load the NQ dataset
dataset = load_dataset("lighteval/natural_questions_clean",cache_dir="./")


extract_top = 10

def split_into_chunks(context, chunk_size=512):
    tokens = context.split()
    chunks = []
    current_chunk = []
    current_count = 0
    
    for token in tokens:
        if current_count + 1 > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_count = 0
        
        current_chunk.append(token)
        current_count += 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def hf_embedding_store(model_name: str) -> CohereEmbedding:
    return HuggingFaceEmbedding(model_name=model_name, device = 'cuda:0',cache_folder="./")

def hf_embedding_ret(model_name: str) -> CohereEmbedding:
    return HuggingFaceEmbedding(model_name=model_name, device = 'cuda:0',cache_folder="./")

store_model = hf_embedding_store(model_name="Salesforce/SFR-Embedding-Mistral")
ret_model = hf_embedding_ret(model_name="Salesforce/SFR-Embedding-Mistral")
    
def retriever(documents, embedding_type="float", model_name="embed-english-v3.0"):
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model= OpenAIEmbedding(),
    )
    return VectorIndexRetriever(
        index=index,
        similarity_top_k=extract_top,
        embed_model = OpenAIEmbedding(),
    )
    
def retrieve_passages(example):
    try:
        query = example["question"]
        content = split_into_chunks(example['document'])
        print(f"Length of Content: {len(content)}")
        # Remove the values in the list that are empty
        content = list(filter(None, content))
        documents = [Document(text=context) for context in content]
        retriever_int8 = retriever(documents, "int8")
        retrieved_docs = retriever_int8.retrieve(query)
        example["retrieved_passages"] = [doc.text for doc in retrieved_docs]
    except Exception as e:
        print(f"Error: {e}")
        example["retrieved_passages"] = []
    return example

eval_dataset = dataset["validation"].map(retrieve_passages, batched=False)
eval_dataset.save_to_disk(f"./NQ_openaiembed_extracted{extract_top}")
