# Datasets

This folder contains datasets used in the ContextualBench project. Below 

## Contents

- [2WikiHotpotQA](#2WikiHotpotQA)
- [HotpotQA](#HotpotQA)
- [MuSiQue](#MuSiQue)
- [NaturalQuestions](#NaturalQuestions)
- [PopQA](#PopQA)
- [TriviaQA](#TriviaQA)
- [TruthfulQA](#TruthfulQA)

## 2WikiHotpotQA

This dataset is a multihop question answering task, as proposed in "Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps" by Ho. et. al 
The folder contains evaluation script and path to dataset on the validation split on around 12k samples. 
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

## HotpotQA

HotpotQA is a Wikipedia-based question-answer pairs with the questions require finding and reasoning over multiple supporting documents to answer. We evaluate on 7405 datapoints, on the distractor setting. This dataset was proposed in the below paper 
```
@inproceedings{yang2018hotpotqa,
  title={{HotpotQA}: A Dataset for Diverse, Explainable Multi-hop Question Answering},
  author={Yang, Zhilin and Qi, Peng and Zhang, Saizheng and Bengio, Yoshua and Cohen, William W. and Salakhutdinov, Ruslan and Manning, Christopher D.},
  booktitle={Conference on Empirical Methods in Natural Language Processing ({EMNLP})},
  year={2018}
}
```

## MuSiQue

This dataset is a multihop question answering task, that requires 2-4 hop in every questions, making it slightly harder task when compared to other multihop tasks.This dataset was proposed in the below paper 

```
@article{trivedi2021musique,
  title={{M}u{S}i{Q}ue: Multihop Questions via Single-hop Question Composition},
  author={Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  journal={Transactions of the Association for Computational Linguistics},
  year={2022}
  publisher={MIT Press}
}
```

## NaturalQuestions

The NQ corpus contains questions from real users, and it requires QA systems to read and comprehend an entire Wikipedia article that may or may not contain the answer to the question

```
@article{47761,
title	= {Natural Questions: a Benchmark for Question Answering Research},
author	= {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le and Slav Petrov},
year	= {2019},
journal	= {Transactions of the Association of Computational Linguistics}
}
```

## PopQA
PopQA is a large-scale open-domain question answering (QA) dataset, the long-tail subset, consisting of 1,399 rare entity queries whose monthly Wikipedia page views are less than 100
Make sure to cite the work
```
@article{ mallen2023llm_memorization ,
  title={When Not to Trust Language Models: Investigating Effectiveness and Limitations of Parametric and Non-Parametric Memories },
  author={ Mallen, Alex and Asai,Akari and  Zhong, Victor and Das, Rajarshi and Hajishirzi, Hannaneh and Khashabi, Daniel},
  journal={ arXiv preprint },
  year={ 2022 }
}
```

## TriviaQA

TriviaqQA is a reading comprehension dataset containing question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents, six per question on average, that provide high quality distant supervision for answering the questions.
```
@article{2017arXivtriviaqa,
       author = {{Joshi}, Mandar and {Choi}, Eunsol and {Weld},
                 Daniel and {Zettlemoyer}, Luke},
        title = "{triviaqa: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension}",
      journal = {arXiv e-prints},
         year = 2017,
          eid = {arXiv:1705.03551},
        pages = {arXiv:1705.03551},
archivePrefix = {arXiv},
       eprint = {1705.03551},
}
```

## TruthfulQA

TruthfulQA is a benchmark to measure whether a language model is truthful in generating answers to questions. The benchmark comprises 817 questions that span 38 categories, including health, law, finance and politics. Questions are crafted so that some humans would answer falsely due to a false belief or misconception. To perform well, models must avoid generating false answers learned from imitating human texts.

```
@misc{lin2021truthfulqa,
    title={TruthfulQA: Measuring How Models Mimic Human Falsehoods},
    author={Stephanie Lin and Jacob Hilton and Owain Evans},
    year={2021},
    eprint={2109.07958},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```