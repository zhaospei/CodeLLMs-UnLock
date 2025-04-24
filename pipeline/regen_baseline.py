import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import os

from sklearn.metrics import (
    classification_report,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
)
import dataeval.w_mbpp as mbpp
import dataeval.w_repoeval as repo_eval
import argparse
import dataeval.w_humaneval as human_eval
import dataeval.w_mbpp as mbpp
import dataeval.w_ds1000 as ds1000
import dataeval.w_repoeval as repo_eval
import random
from torch.utils.data import DataLoader


random.seed(42)
DATASET_ROOT= os.path.join(_settings.DATA_FOLDER, "MBPP", "data")

continue_re_gen_from = 'data_no_ignore/test_false_codellama_mbpp_re_gen.parquet'
df = pd.read_parquet(continue_re_gen_from)
output_gen_dict = {}
regen_task_id_list = df["task_id"].unique().tolist()

for _, row in df.iterrows():
    if row['task_id'] not in output_gen_dict:
        output_gen_dict[row['task_id']]  = []
    output_gen_dict[row['task_id']].append({
        'generated_code': row['extracted_code'],
        'task_id': row['task_id'],
        'completion_id': row['completion_id'],
        'label': row['label'],
    })

    
def read_test_examples(data_path: str):
    def format_test_example(q, tests, code: str=None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    examples = [json.loads(x) for x in open(data_path)]
    print("Read all {} examples from {} over!".format(len(examples), data_path))

    # test_cases
    examples_str = []
    for i in range(1, 4):
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        ex_prompt = format_test_example(q, test, code)
        example_prompt = '- Example {}:\n{}'.format(i, ex_prompt)
        examples_str += [example_prompt]

    for i in range(10, 510):
        if i not in regen_task_id_list:
            continue
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        
        prompt = format_test_example(q, test, code=None)

        for gen_dict in output_gen_dict[ex['task_id']]:
            prompt_with_shots = '''
Please refer the given examples and generate a python function for my problem.\n
Examples are listed as follows:
{}

Here is my problem:
{}

Here is my generated code:
{}

My generated code is not correct. Please fix it.
'''.strip().format('\n\n'.join(examples_str), prompt, gen_dict['generated_code'])
            yield {
                "task_id": ex["task_id"],
                "generated_code": gen_dict['generated_code'],
                "prompt": prompt_with_shots,
                "label": gen_dict['label'],
                "completion_id": gen_dict['completion_id'],
            }




def get_model_tokenize(model_name):
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name, resume_download=True, trust_remote_code=True
    # )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    # model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Need to set the padding token to the eos token for generation
    if tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "left"
    return model, tokenizer

def eval_prompt(args):
    # done_prompt
    model, tokenizer = get_model_tokenize(args.model)

    model_name = args.model.replace("/", "-")
    generation_config = {"do_sample": False, "max_new_tokens": 512, "num_beams": 1}

    generated_texts = []

    batch_size = 4

    problem_file = os.path.join(DATASET_ROOT, f"mbpp.jsonl")
    examples = list(read_test_examples(problem_file))
    
    # Convert prompts into a DataLoader for batching
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": item["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for item in examples
    ]
    # print(prompts[0])
    data_loader = DataLoader(prompts, batch_size=batch_size, shuffle=False)

    for batch in tqdm(data_loader, desc="Generating in batches"):
        inputs = tokenizer(batch, return_tensors="pt", padding="longest").to("cuda")
        # print(inputs)
        generation_config["pad_token_id"] = tokenizer.eos_token_id
        generated_ids = model.generate(**inputs, **generation_config)

        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for i_, source in enumerate(batch):
            res = output[i_]
            generated_texts.append(res)

    for example, res in zip(examples, generated_texts):
        example["generated_code"] = res
        
    answer_df = pd.DataFrame(examples)
    answer_df.to_parquet(continue_re_gen_from.replace('.parquet', '-answers.parquet'))
    print(f"Saved to {continue_re_gen_from.replace('.parquet', '-answers.parquet')}")
    return answer_df
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/deepseek-coder-1.3b-instruct",
        help="which results to run",
    )
    args = parser.parse_args()
    
    eval_prompt(args)
