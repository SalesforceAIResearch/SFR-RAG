# TriviaQA

## Overview
TriviaqQA is a reading comprehension dataset containing question-answer pairs authored by trivia enthusiasts and independently gathered evidence documents, six per question on average, that provide high quality distant supervision for answering the questions. 

This setting can also be found in the ContextualBench repository (INCLUDE_LINK) and can be loaded using the following code
```
load_dataset("Salesforce/ContextualBench","triviaqa",split="validation")
```

This dataset was proposed in the paper 
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