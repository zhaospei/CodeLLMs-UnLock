from datasets import load_dataset
import pandas as pd
import datasets
from datasets import Dataset
import os
import _settings
import json
DATASET_ROOT= os.path.join(_settings.DATA_FOLDER, "RepoExec", "datasets")
STOP_WORDS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>"]

TEMPLATE = """\
Please complete the {function_name} function based on the contexts above the function.

The contexts above the function are:
```Python
{contexts_above}
```

The code to be completed is:
```Python
{input_code}
```

Completed code:
"""

def load_jsonl(fname):
    with open(fname, 'r', encoding='utf8') as f:
        lines = []
        for line in f:
            lines.append(json.loads(line))
        return lines


def _save_dataset(tokenizer, language, max_seq_len, max_gen_len, sft=False, instruction=False):
    save_path = f"{DATASET_ROOT}/{language}" if not sft else f"{DATASET_ROOT}/{language}_sft"
    save_path = f"{save_path}_instruction" if instruction else save_path
    
    # if not os.path.exists(save_path):
    data_path = os.path.join(DATASET_ROOT, "small_context-00000-of-00001.parquet")
    # lines = load_jsonl(data_path)
    lines = pd.read_parquet(data_path).to_dict(orient='records')
    dataset = {}
    dataset["prompt"] = []
    dataset["task_id"] = []
    dataset["original_prompt"] = []
    dataset["canonical_solution"] = []
    dataset["stopwords"] = []
    if instruction:
        for idx, sample in enumerate(lines):
            contexts_above = sample['prompt'].replace(sample['target_function_prompt'].strip(), "")
            input_code = sample["function_signature"][:-1].replace("\\", "").replace("\n", "")
            input_ids = tokenizer.encode(input_code)
            max_context_length = max_seq_len - len(input_ids) - max_gen_len
            context_above_ids = tokenizer.encode(contexts_above)
            # context_above = sample['contexts_above']
            if len(context_above_ids) > max_context_length:
                context_above_ids = context_above_ids[-max_context_length:]
                contexts_above = tokenizer.decode(context_above_ids)
            prompt = TEMPLATE.format(
                function_name=sample['entry_point'],
                contexts_above=contexts_above,
                input_code=input_code,
            )
            dataset["prompt"].append(prompt)
            dataset["task_id"].append(sample["id"])
            dataset["original_prompt"].append(sample["prompt"])
            dataset["canonical_solution"].append(sample["solution"])
            dataset["stopwords"].append(STOP_WORDS)
    else:
        for idx, sample in enumerate(lines):
            contexts_above = sample['prompt'].replace(sample['target_function_prompt'].strip(), "")
            input_code = sample["function_signature"][:-1].replace("\\", "").replace("\n", "")
            input_ids = tokenizer.encode(input_code)
            max_context_length = max_seq_len - len(input_ids) - max_gen_len
            context_above_ids = tokenizer.encode(contexts_above)
            if len(context_above_ids) > max_context_length:
                context_above_ids = context_above_ids[-max_context_length:]
                contexts_above = tokenizer.decode(context_above_ids)
            prompt = contexts_above + "\n" + input_code
            dataset["prompt"].append(prompt)
            dataset["task_id"].append(sample["id"])
            dataset["original_prompt"].append(sample["prompt"])
            dataset["canonical_solution"].append(sample["solution"])
            dataset["stopwords"].append(STOP_WORDS)
        
    data_df = pd.DataFrame.from_dict(dataset)
    dataset = Dataset.from_pandas(data_df)
    
    dataset.save_to_disk(save_path)
    return save_path

# _save_dataset(sft=False)
def get_dataset(tokenizer, language, sft=False, instruction=False):
    dataset = datasets.load_from_disk(_save_dataset(tokenizer, language, max_seq_len=2048, max_gen_len=1024, sft=sft, instruction=instruction))

    def encode_humaneval(example):
        prompt = example['prompt']
        if instruction:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": f'{prompt}'}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, truncation=False, padding=False)
        return inputs

    dataset = dataset.map(encode_humaneval, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)

    return dataset

def postprocess_by_function(generation, target):
    first_token = target.split()[0]
    function_indent = target.split(first_token)[0]
    generation_lines = []
    for line in generation.split('\n'):
        if line.split() and line.split()[0]!='#':
            first_token = line.split()[0]
            indent = line.split(first_token)[0]
            if len(indent) < len(function_indent):
                break
            generation_lines.append(line)
    return generation_lines