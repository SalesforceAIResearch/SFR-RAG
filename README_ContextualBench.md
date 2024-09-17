
# ContextualBench - A comprehensive toolkit to evaluate LM on different Contextual datasets

## Description

ContextualBench is a powerful evaluation framework designed to assess the performance of Large Language Models (LLMs) on contextual datasets. It provides a flexible pipeline for evaluating various LLM families across different tasks, with a focus on handling large context inputs.

> Each individual evaluation dataset in ContextualBench is licensed separately and must be adhered by a user.


## Features

Dynamic Retrieval Support: Efficiently handles large context inputs, allowing for comprehensive evaluation of LLMs' contextual understanding capabilities.
Extensive Evaluation Dataset: Supports 7 contextual tasks, including: Question Answering (QA), Multi-Hop Question Answering, Classification tasks
Multi-LLM Family Support: Compatible with a wide range of LLM families, including: Hugging Face models, Gemma, Mistral, OpenAI, Cohere.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Cite](#cite)

## Environment Setup

Install the requirement.txt file using a conda environment having atleast python 3.9 and above using the command
```
conda create --name contextualbench python=3.9
conda activate contextualbench
pip install -r requirements.txt
```

Make sure to follow the steps in the [vllm_flashinfer_steps.txt](./vllm_flashinfer_steps.txt) file to install the latest transformer, vllm and flashinfer (depending on your system) versions
```
pip install transformers --upgrade
pip install vllm==v0.5.3.post1
```

## Usage

Use the [`config/config.yaml`](./config/config.yaml) file to tweak the hyperparameters like temperature, max_tokens, top_p etc. also initialise the different API keys that might be needed.

### Getting contextual dataset
Existing retrieved context uploaded on [ContextualBench](https://huggingface.co/datasets/Salesforce/ContextualBench) can be used, or custom dataset can be generated using the `retriever.py` files present in the respective folder. 
Once the data is ready, make sure you have all the required libraries installed (and also environment variable set for OPENAI_API_KEY and COHERE_API_KEY)

### Evaluating models on the contextual dataset
#### Method 1
Simply execute the command specified in the respective dataset folder to evaluate.

#### Method 2
You can also use the `run.py` file to run any dataset (except PopQA, use the command mentioned in the PopQA directory)
Run the command
```bash
python run.py [dataset_name]
```
Here [dataset_name] can be replaced by 2wikimultihopqa, hotpotqa, musique, naturalquestions, triviaqa, truthfulqa.


## Leaderboard
We manage a leaderboard that ranks large language models (LLMs) based on their performance on ContextualBench Tasks, which can be found at the [ContextualBench Leaderboard](https://huggingface.co/spaces/Salesforce/RAG-Leaderboard). To have your model evaluated and included on the leaderboard, please send your model's predictions (outputs) for all datasets to [xnguyen@salesforce.com](xnguyen@salesforce.com).

## Cite
If you use this work, please cite the following -


```
@article{nguyen2024sfrrag,
  title={SFR-RAG: Towards Contextually Faithful LLMs},
  author={Nguyen, Xuan-Phi and Pandit, Shrey and Purushwalkam, Senthil and Xu, Austin and Chen, Hailin and Ming, Yifei and Ke, Zixuan and Savarese, Silvio and Xong, Caiming and Joty, Shafiq},
  year={2024}
}

```


