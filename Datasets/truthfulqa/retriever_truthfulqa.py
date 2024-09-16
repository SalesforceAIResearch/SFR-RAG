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
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from datasets import load_dataset, Dataset
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import requests
import json 


gen_dataset = load_dataset("truthful_qa", "generation")[
                "validation"
            ]
mc_dataset = load_dataset("truthful_qa", "multiple_choice")[
    "validation"
]
df_mc, df_gen = mc_dataset.to_pandas(), gen_dataset.to_pandas()
merged_df = pd.merge(
    df_mc,
    df_gen[["question", "category", "source"]],
    on="question",
    how="left",
)
mc_dataset = Dataset.from_pandas(merged_df)

def get_text_from_website(data):
    link = data['source']
    print(link)
    try:
        if "#" in link:
            url = link.split("#")[0]
            section_id = link.split("#")[1]
            # Fetch the content of the page
            response = requests.get(url,timeout=60)
            if response.status_code != 200:
                # return f"Failed to retrieve the page. Status code: {response.status_code}"
                raise Exception(f"Failed to retrieve the page. Status code: {response.status_code}")

            # Parse the HTML content
            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the section header with the specified id
            section_span = soup.find(id=section_id)
            if not section_span:
                # return f"Section with id '{section_id}' not found."
                raise Exception(f"Section with id '{section_id}' not found.")

            # Find the parent heading tag (like <h3>)
            section_header = section_span.find_parent(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if not section_header:
                # return f"Heading for section with id '{section_id}' not found."
                raise Exception(f"Heading for section with id '{section_id}' not found.")
            

            # Determine the level of the current section header
            header_level = int(section_header.name[1])

            # Find all elements after the section header until the next section header of the same level or higher
            content = []
            for sibling in section_header.find_all_next():
                if sibling.name and sibling.name.startswith('h') and int(sibling.name[1]) <= header_level:
                    break
                content.append(sibling.get_text(separator="\n").strip())
            text = "\n".join(content)
            
        else:
            response = requests.get(link,timeout=60)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text()

    except requests.exceptions.Timeout:
        print("Error: The request timed out.")
        text = " "
            
    except Exception as e:
        print(f"Error: {e}")
        text = " "
        
    cleaned_text = re.sub(r'\[\d+\]', '', text.strip())
    cleaned_text = re.sub(r'\n', ' ', cleaned_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    data['website_data'] = cleaned_text
    return data

# Extract the section content

print("Getting text from website")
mc_dataset_text = mc_dataset.map(get_text_from_website)

extract_top = 10

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
    query = example["question"]
    list_docs = example["website_data"].split(". ")
    documents = [Document(text=context) for context in list_docs]
    try:
        retriever_int8 = retriever(documents, "int8")
        retrieved_docs = retriever_int8.retrieve(query)
        example["retrieved_passages"] = [doc.text for doc in retrieved_docs]
    except Exception as e:
        print(f"Error: {e}")
        example["retrieved_passages"] = []
    return example

print("Retrieving passages")
eval_dataset = mc_dataset_text.map(retrieve_passages, batched=False)
eval_dataset.save_to_disk(f"./truthfulQA_val_SFR2Rembed{extract_top}_size{len(eval_dataset)}")