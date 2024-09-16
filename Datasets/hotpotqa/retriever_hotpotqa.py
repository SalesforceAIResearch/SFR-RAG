from datasets import load_dataset, Dataset
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.embeddings.cohere import CohereEmbedding
import os
from tqdm import tqdm
import cohere
import time
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import requests
import json 

# Importing the keys and setting the environment variables


# Load the HotpotQA dataset
dataset = load_dataset("hotpot_qa","distractor", cache_dir="./")

def get_doc(text):
    documents = []
    titles = text['title']
    sentences = text['sentences']
    for title, sentence_list in zip(titles, sentences):
        combined_text = title + ' ' + ' '.join(sentence_list)
        documents.append(combined_text)
    return documents

# Process each split of the dataset separately
train_documents = [get_doc(text) for text in tqdm(dataset['validation']['context'])]
dataset['validation'] = dataset['validation'].add_column('rag', train_documents)

extract_top = 15

def hf_embedding_store(model_name: str) -> CohereEmbedding:
    return HuggingFaceEmbedding(model_name=model_name, device = 'cuda:0',cache_folder="./")

def hf_embedding_ret(model_name: str) -> CohereEmbedding:
    return HuggingFaceEmbedding(model_name=model_name, device = 'cuda:0',cache_folder="./")

store_model = hf_embedding_store(model_name="Salesforce/SFR-Embedding-2_R")
ret_model = hf_embedding_ret(model_name="Salesforce/SFR-Embedding-2_R")
    
def retriever(documents, embedding_type="float", model_name="embed-english-v3.0"):
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model= store_model,
    )
    return VectorIndexRetriever(
        index=index,
        similarity_top_k=extract_top,
        embed_model = ret_model,
    )
    
def retrieve_passages(example):
    try:
        query = example["question"]
        documents = [Document(text=context) for context in example['rag']]
        retriever_int8 = retriever(documents, "int8")
        retrieved_docs = retriever_int8.retrieve(query)
        example["retrieved_passages"] = [doc.text for doc in retrieved_docs]
    except Exception as e:
        print(f"Error: {e}")
        example["retrieved_passages"] = []
    return example

eval_dataset = dataset["validation"].map(retrieve_passages, batched=False)
eval_dataset.save_to_disk(f"./hotpot_val_SFR-2R_extracted{extract_top}_size{len(eval_dataset)}")