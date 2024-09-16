# 2wikimultihop Dataset

## Overview

The 2wikimultihop dataset is designed for multi-hop question answering tasks. It contains questions that require reasoning across multiple Wikipedia articles to find the correct answer. This dataset aims to challenge and improve the capabilities of question answering systems in handling complex, multi-step reasoning tasks. We evaluate on 12,576 datapoints of validation set.

## Running the Code

Make sure to install relevant dependencies:
```
pip install transformers --upgrade
pip install vllm==v0.5.3.post1
```
Export the relevant keys in the environment variable and also in the `config.yaml` file in the repository:
```
export OPENAI_API_KEY = YOUR_OPENAI_KEY
```
To process and analyze the dataset, use the `2wiki.py` script:

```bash
python 2wiki.py
```

## Citation

If you use this dataset in your research, please cite:

```
@inproceedings{xanh2020_2wikimultihop,
    title = "Constructing A Multi-hop {QA} Dataset for Comprehensive Evaluation of Reasoning Steps",
    author = "Ho, Xanh  and
      Duong Nguyen, Anh-Khoa  and
      Sugawara, Saku  and
      Aizawa, Akiko",
    booktitle = "Proceedings of the 28th International Conference on Computational Linguistics",
    month = dec,
    year = "2020",
    address = "Barcelona, Spain (Online)",
    publisher = "International Committee on Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.coling-main.580",
    pages = "6609--6625",
}
```