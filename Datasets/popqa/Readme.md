# PopQA

## Overview
PopQA is a large-scale open-domain question answering (QA) dataset, the long-tail subset, consisting of 1,399 rare entity queries whose monthly Wikipedia page views are less than 100

### Running the code
To Run the PopQA task, replace the run_baseline.py file in the self_rag/retrieval_lm folder of the [self-rag repository](https://github.com/AkariAsai/self-rag/tree/main)
Then follow the steps provided in the self-rag codebase to run the code.
Make sure to install relevant dependencies:
```
pip install transformers --upgrade
pip install vllm==v0.5.3.post1
```
Export the relevant keys in the environment variable and also in the config.yaml file in the repository:
```
export OPENAI_API_KEY = YOUR_OPENAI_KEY
```
Followed by running th command:
```
python run_baseline_lm.py \
--model_name "gpt-4o-mini" \
--input_file popqa_longtail_w_gs.jsonl \
--max_new_tokens 100 --metric match \
--result_fp ./popqa \
 --task qa \
--mode retrieval \
--world_size 1 \
--prompt_name "prompt_no_input_retrieval"
```

### Citation
Make sure to cite the work
```
@article{ mallen2023llm_memorization ,
  title={When Not to Trust Language Models: Investigating Effectiveness and Limitations of Parametric and Non-Parametric Memories },
  author={ Mallen, Alex and Asai,Akari and  Zhong, Victor and Das, Rajarshi and Hajishirzi, Hannaneh and Khashabi, Daniel},
  journal={ arXiv preprint },
  year={ 2022 }
}
```