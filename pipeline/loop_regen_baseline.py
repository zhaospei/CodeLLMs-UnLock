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
        batch_inputs = tokenizer([prompt], truncation=False, padding=False)
        input_ids = batch_inputs['input_ids'].to(device)
        attention_masks = batch_inputs['attention_mask'].to(device)
        dict_outputs =  model.generate(input_ids, attention_masks,
            num_beams=1, max_new_tokens=args.max_new_tokens, 
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            output_hidden_states = True, return_dict_in_generate=True, output_scores=True
        )
        print(dict_outputs.keys())
        hidden_states = dict_outputs.hidden_states
        print(hidden_states[0][0].shape)
        #     generation = dict_outputs.sequences[:, input_length:].cpu()
        #     # print(f"Generation shape: {generation.shape}")
        #     for gen in generation:
        #         generations.append(gen)
            
        #     layers_to_process = args.layers
        #     hidden_states = dict_outputs.hidden_states
        #     ###### hidden_states : (num_tokens, num_layers, num_seq, num_input_tokens/1, embedding_size)
            
        #     for layer in layers_to_process:
        #         all_token_hidden_states_layer = {}
        #         for ind in range(hidden_states[1][-1].shape[0]):
        #             all_token_hidden_states_layer[ind + off_set] = []
        #             for hidden_state in hidden_states[1:]:
        #                 all_token_hidden_states_layer[ind + off_set].append(hidden_state[layer][ind, -1, :].detach().cpu().float().numpy())

        #         if layer not in all_token_hidden_states_layer_list:
        #             all_token_hidden_states_layer_list[layer] = {}
        #         all_token_hidden_states_layer_list[layer].update(all_token_hidden_states_layer)
        #     # return hidden_state
                
        # generation_config["pad_token_id"] = tokenizer.eos_token_id
        # generated_ids = model.generate(**inputs, **generation_config)
        # generated_ids = generated_ids[inputs["input_ids"].shape[1]:].cpu()
        # output = tokenizer.decode(generated_ids, skip_special_tokens=True)
        # print(output)

        return output
    problem_file = os.path.join(DATASET_ROOT, f"mbpp.jsonl") 
    mbpp_re_gen_prompt_dict = read_test_examples(problem_file)
    
    
    embedding_dim = 4096
    clf_model = BinaryClassifier(embedding_dim)
    clf_model.load_state_dict(torch.load(args.clf_model_path))
    clf_model.to("cuda")
    clf_model.eval()
    
    continue_re_gen_from = args.file_path
    df = pd.read_parquet(continue_re_gen_from)


    for _, row in df.iterrows():
        embedding = torch.tensor(row['last_token_code_embedding']).float().to("cuda")
        with torch.no_grad():
            output = clf_model(embedding)
            pred = (output > 0.5).float().item()
            if pred == 0.0:
                output = eval_prompt(row['task_id'], row['generated_code'])
                print("Regenerated code for task_id:", row['task_id'])
                print("Generated code:", row['generated_code'])
                print("Regen code:", output)   
    
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
