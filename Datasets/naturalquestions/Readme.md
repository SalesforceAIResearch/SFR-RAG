# Natural Questions

## Overview
The NQ corpus contains questions from real users, and it requires QA systems to read and comprehend an entire Wikipedia article that may or may not contain the answer to the question. We evaluate the models on 7830 validation datapoints.

## Extracting the top k context
We use the "RAG" column of the dataset that is concatenation of the different contexts that can be used in the RAG based setting task. Set the variables of extract_top and embedding versions (defaulted to SFR-2R Embeddings) and run the command
```
python retriever_NQ.py
```
to generate the dataset with top_k context, This setting can also be found in the [ContextualBench repository](https://huggingface.co/datasets/Salesforce/ContextualBench) and can be loaded using the following code
```
load_dataset("Salesforce/ContextualBench","NaturalQuestions",split="validation")
```

## Running the code
Make sure to install relevant dependencies:
```
pip install transformers --upgrade
pip install vllm==v0.5.3.post1
```
Export the relevant keys in the environment variable and also in the config.yaml file in the repository:
```
export OPENAI_API_KEY = YOUR_OPENAI_KEY
```

Then finally run the command
```
python NQ.py
```

## Citation
This dataset was proposed in the below paper and should be citated as:
```
@article{47761,
title	= {Natural Questions: a Benchmark for Question Answering Research},
author	= {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le and Slav Petrov},
year	= {2019},
journal	= {Transactions of the Association of Computational Linguistics}
}
```