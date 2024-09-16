# HotpotQA

## Overview
HotpotQA is a Wikipedia-based question-answer pairs with the questions require finding and reasoning over multiple supporting documents to answer. We evaluate on 7405 datapoints, on the distractor setting. 
An example of datapoint is 
```
{
    "answer": "This is the answer",
    "context": {
        "sentences": [["Sent 1"], ["Sent 21", "Sent 22"]],
        "title": ["Title1", "Title 2"]
    },
    "id": "000001",
    "level": "medium",
    "question": "What is the answer?",
    "supporting_facts": {
        "sent_id": [0, 1, 3],
        "title": ["Title of para 1", "Title of para 2", "Title of para 3"]
    },
    "type": "comparison",
    "RAG": "Context"
}
```

## Extracting the top k context
We use the "RAG" column of the dataset that is concatenation of the different contexts that can be used in the RAG based setting task. Set the variables of extract_top and embedding versions (defaulted to SFR-2R Embeddings) and run the command
```
python retriever_hotpotqa.py
```
to generate the dataset with top_k context, This setting can also be found in the [ContextualBench repository](https://huggingface.co/datasets/Salesforce/ContextualBench) and can be loaded using the following code
```
load_dataset("Salesforce/ContextualBench","hotpotqa",split="validation")
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
python hotpotqa.py
```

## Citation
This dataset was proposed in the below paper 
```
@inproceedings{yang2018hotpotqa,
  title={{HotpotQA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William W. and Salakhutdinov, Ruslan and Manning, Christopher D.},
  booktitle={Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  year={2018}
}
```