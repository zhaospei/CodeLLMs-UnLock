import pandas as pd
import argparse
import os
import dataeval.w_humaneval as human_eval
import dataeval.w_mbpp as mbpp
import dataeval.w_ds1000 as ds1000
import dataeval.w_repoeval as repo_eval
import dataeval.w_deveval as dev_eval
import models
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
from dataeval.w_humaneval import extract_generation_code as human_eval_egc
from dataeval.w_mbpp import extract_generation_code as mbpp_eval_egc
from dataeval.w_ds1000 import extract_generation_code as ds1000_eval_egc
from dataeval.w_evocodebench import extract_generation_code as evocodebench_eval_egc
from dataeval.w_repoeval import extract_generation_code as repoeval_eval_egc
from dataeval.w_deveval import extract_generation_code as deveval_eval_egc


PROMPT_TEMPLATE = """
Evaluate the following code outputs. For each coding problem, you will be given an LLM-generated code output. You must determine if the LLM-generated code output is correct or not. If the LLM-generated code output is correct, write '1'; if it is not correct, write '0'.
Problem: %s
LLM Output: %s
Correctness:
"""



def get_dataset_fn(data_name):
    if data_name == 'human_eval':
        return human_eval.get_dataset
    if data_name == 'mbpp':
        return mbpp.get_dataset
    if data_name == 'ds1000':
        return ds1000.get_dataset
    if data_name == 'repo_eval':
        return repo_eval.get_dataset
    if data_name == 'dev_eval':
        return dev_eval.get_dataset

def get_prompt_dataset(dataset):
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-base", trust_remote_code=True)
    datasets = get_dataset_fn(dataset)(tokenizer, language='python', instruction=False)
    meta_dataset = dict()
    for task in datasets:
        task_id = task['task_id']
        prompt = task['original_prompt']
        task = dict(task)
        task['prompt'] = prompt
        meta_dataset[task_id] = task
    return meta_dataset


def extract_generation_code_fun(data_name):
    if data_name == 'human_eval':
        return human_eval_egc
    if data_name == 'mbpp':
        return mbpp_eval_egc
    if data_name == 'ds1000':
        return ds1000_eval_egc
    if data_name == 'repo_eval':
        return repoeval_eval_egc
    if data_name == 'evocodebench':
        return evocodebench_eval_egc
    if data_name == 'repoexec':
        return repoeval_eval_egc
    if data_name == 'dev_eval':
        return deveval_eval_egc

def get_generation_code(args,example,generation):
    ds_fn = extract_generation_code_fun(args.dataset)
    return ds_fn(example,generation,'python')

def main(args):
    datasets = get_prompt_dataset(args.dataset)
    df = pd.read_parquet(args.file)
    results = list()
    for i, row in df.iterrows():
        task_id = row['task_id']
        data = datasets[task_id]
        api_id = row['completion_id']
        if  hasattr(row,'cleaned_code'):
            llm_out = row['cleaned_code']
        elif hasattr(row,'extracted_code'):
            llm_out = row['extracted_code']
        else:
            llm_out = get_generation_code(args,data,row['generation'])
        prompt = PROMPT_TEMPLATE.strip() % (data['prompt'], llm_out)
        post_api = {"custom_id":api_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                    "model": "o4-mini-2025-04-16",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 1,
                    "max_completion_tokens": 2048}}
        results.append(post_api)
    with open(f'openai/{args.output}/batch_{args.model}_{args.dataset}.jsonl','w+') as f:
        for el in results:
            f.writelines(json.dumps(el)+'\n')

if __name__ == "__main__":
    # Load the DataFrame from a Parquet file
    args = argparse.ArgumentParser()
    args.add_argument("--file", type=str)
    args.add_argument("--output", type=str)
    args.add_argument("--dataset", type=str )
    args.add_argument("--model", type=str)
    args = args.parse_args()
    print(args)
    main(args)