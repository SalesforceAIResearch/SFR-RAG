# MuSiQue

## Overview
This dataset is a multihop question answering task, that requires 2-4 hop in every questions, making it slightly harder task when compared to other multihop tasks. 

## Running the Code
This dataset with context can also be found in the [ContextualBench repository](https://huggingface.co/datasets/Salesforce/ContextualBench) and can be loaded using the following code
```
load_dataset("Salesforce/ContextualBench","MuSiQue",split="validation")
```

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
python MusiQue.py
```

## Citation
This dataset was proposed in the below paper and should be cited as: 
```
@article{trivedi2021musique,
  title={{M}u{S}i{Q}ue: Multihop Questions via Single-hop Question Composition},
  author={Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  journal={Transactions of the Association for Computational Linguistics},
  year={2022}
  publisher={MIT Press}
}
```