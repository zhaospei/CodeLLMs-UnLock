import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import os
import argparse
import random
from torch.utils.data import DataLoader
# import _settings
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import dataeval.w_mbpp as mbpp
from dataeval.w_mbpp import extract_generation_code as mbpp_eval_egc
from func.metric import *

torch.manual_seed(42)

class BinaryClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))           
        return x

random.seed(42)
DATASET_ROOT= 'benchmark/MBPP/data'

    
def read_test_examples(data_path: str):
    def format_test_example(q, tests, code: str=None):
        prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
        if code:
            code = code.replace("\r", "").replace("\t", "    ")
            prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
        return prompt

    results = {}
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
        ex = examples[i]
        q, test, code = ex['text'], ex['test_list'], ex['code']
        
        prompt = format_test_example(q, test, code=None)

       
        prompt_with_shots = '''
Please refer the given examples and generate a python function for my problem.\n
Examples are listed as follows:
{}

Here is my problem:
{}

Here is my generated code:
<GENERATE_CODE_PLACEHOLDER>

My generated code is not correct. Please fix it.
'''.strip().format('\n\n'.join(examples_str), prompt)
        results[ex["task_id"]] = prompt_with_shots
    return results




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


def main(args):
    model, tokenizer = get_model_tokenize(args.model)
    model_name = args.model.replace("/", "-")
    device = args.device
    def eval_prompt(task_id, generated_code):
        generation_config = {"do_sample": False, "max_new_tokens": 512, "num_beams": 1}
        prompt = mbpp_re_gen_prompt_dict[task_id]
        prompt = prompt.replace("<GENERATE_CODE_PLACEHOLDER>", generated_code)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": f'{prompt}'}], tokenize=False, add_generation_prompt=True)
        batch_inputs = tokenizer([prompt], truncation=False, padding=False, return_tensors='pt').to(device)
        input_ids = batch_inputs['input_ids']
        # print(input_ids)
        attention_masks = batch_inputs['attention_mask']
        # print(attention_masks)
        dict_outputs =  model.generate(**batch_inputs,
            num_beams=1, max_new_tokens=512, 
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            output_hidden_states = True, return_dict_in_generate=True, output_scores=True
        )
        print(dict_outputs.keys())
        hidden_states = dict_outputs.hidden_states
        input_length = input_ids.shape[1]
        ###### hidden_states : (num_tokens, num_layers, num_seq, num_input_tokens/1, embedding_size)
        generated_ids = dict_outputs.sequences[:, input_length:].cpu()[0]
        gen = tokenizer.decode(generated_ids, skip_special_tokens=True)
        clean_generation_decoded = mbpp_eval_egc({'task_id': 0}, gen, 'python')
        start_ind, end_ind = getCleanGenerationRange(generated_ids.tolist(), clean_generation_decoded, tokenizer)
        if start_ind is None or end_ind is None:
            start_ind, end_ind = getGenerationRange(generated_ids.tolist(), tokenizer)
        last_code_token_last_layer_embedding = hidden_states[end_ind - 1][-1][0, -1, :].detach().cpu().float().numpy()

        return gen, last_code_token_last_layer_embedding
    
    problem_file = os.path.join(DATASET_ROOT, f"mbpp.jsonl") 
    mbpp_re_gen_prompt_dict = read_test_examples(problem_file)
    
    
    embedding_dim = 4096
    clf_model = BinaryClassifier(embedding_dim)
    clf_model.load_state_dict(torch.load(args.clf_model_path))
    clf_model.to("cuda")
    clf_model.eval()
    
    continue_re_gen_from = args.file_path
    df = pd.read_parquet(continue_re_gen_from)

    # df = df.sample(n=10)
    for loop_id in range(args.num_loops):
        print("Loop ID:", loop_id)
        predictions = []
        regen_codes = []
        embeddings = []
        
        with tqdm(total=len(df)) as pbar:
            for _, row in df.iterrows():
                if loop_id == 0:
                    embedding = torch.tensor(row['last_token_code_embedding']).float().to("cuda")
                else:
                    if row[f'predictions_{loop_id}'] == 0.0:
                        embedding = torch.tensor(row[f'last_token_code_embedding_{loop_id}']).float().to("cuda")
                    else:
                        print("No need to regenerate for task_id:", row['task_id'])
                        predictions.append(1.0)
                        regen_codes.append("")
                        embeddings.append(None)
                        continue
                with torch.no_grad():
                    output = clf_model(embedding)
                    pred = (output > 0.5).float().item()
                    if pred == 0.0:
                        output, last_code_token_last_layer_embedding = eval_prompt(row['task_id'], row['extracted_code'])
                        print("Regenerated code for task_id:", row['task_id'])
                        print("Generated code:", row['extracted_code'])
                        print("Regen code:", output)
                        predictions.append(pred)
                        regen_codes.append(output)
                        embeddings.append(last_code_token_last_layer_embedding)
                    else:
                        print("No need to regenerate for task_id:", row['task_id'])
                        predictions.append(pred)
                        regen_codes.append("")
                        embeddings.append(None)
                pbar.update(1)
        df[f"predictions_{loop_id + 1}"] = predictions
        df[f"regen_codes_{loop_id + 1}"] = regen_codes
        df[f"last_token_code_embedding_{loop_id + 1}"] = embeddings
    
    df.to_parquet(continue_re_gen_from.replace(".parquet", f"_loop_{args.num_loops}_full.parquet"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/deepseek-coder-1.3b-instruct",
        help="which results to run",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mbpp",
        help="which dataset to use",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default="data_no_ignore/test_false_codellama_mbpp_re_gen.parquet",
        help="continue re-gen from",
    )
    parser.add_argument(
        "--clf_model_path",
        type=str,
        help="classifier model path"
    )
    parser.add_argument(
        "--num_loops",
        type=int,
        default=1,
        help="number of loops to regenerate",
    )
    
    args = parser.parse_args()
    
    main(args)
