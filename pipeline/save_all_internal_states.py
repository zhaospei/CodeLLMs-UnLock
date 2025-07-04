import argparse
import glob
import json
import os
import copy
import time
import gc
import pandas as pd
import numpy as np
import torch
import tqdm
import pickle
from transformers import StoppingCriteria, StoppingCriteriaList
from sentence_transformers import SentenceTransformer
from torchmetrics.text.bert import BERTScore

import _settings
import dataeval.w_humaneval as human_eval
import dataeval.w_mbpp as mbpp
import dataeval.w_ds1000 as ds1000
import dataeval.w_repoeval as repo_eval
import dataeval.w_evocodebench as evocodebench
import dataeval.w_repoexec as repo_exec
import dataeval.w_deveval as dev_eval
from dataeval.w_humaneval import cleanup_code as human_eval_cleanup_code
import models
import utils
from func.metric import *

passed_input_len_task = ['repo_eval', 'evocodebench', 'repoexec']

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-13b-hf')
parser.add_argument('--dataset', type=str, default='human_eval')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--tensor_parallel_size', type=int, default=1)
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--num_generations_per_prompt', type=int, default=10)
parser.add_argument('--max_num_gen_once', type=int, default=10)
parser.add_argument('--max_new_tokens', type=int, default=500)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--decoding_method', type=str, default='greedy')
parser.add_argument('--top_p', type=float, default=0.99)
parser.add_argument('--top_k', type=int, default=10)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--nprocess', type=int, default=None)
parser.add_argument('--project_ind', type=int, default=0)
parser.add_argument("--layers", default=-1, nargs='*', type=int,
                        help="List of layers of the LM to save embeddings from indexed negatively from the end")
parser.add_argument("--language", default="python", type=str,)
parser.add_argument("--load_in_8bit", action="store_true", help="Whether to load the model in 8bit mode")

args = parser.parse_args()
print(args)
print(args.model.replace('/', '_'))
ml_time = int(time.time() * 1000)
layer_name = '_'.join(str(x) for x in args.layers)
if len(args.layers) > 10:
    layer_name = 'all_layers'

OUTPUT_DIR = os.path.join(_settings.GENERATION_FOLDER, f'ALL_llm_check_{args.model.replace("/", "_")}_{args.dataset}_{args.language}_{layer_name}')
os.makedirs(OUTPUT_DIR, exist_ok=True)
logInfo = open(os.path.join(OUTPUT_DIR, "logInfo.txt"), mode="w",encoding="utf-8")

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_str, tokenizer):
        StoppingCriteria.__init__(self)
        self.current_context = []
        self.tokenizer = tokenizer
        self.keywords_str = keywords_str
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        self.current_context.append(input_ids[0][-1].item())
        current_context = self.tokenizer.decode(self.current_context)
        for word in self.keywords_str:
            if word in current_context:
                return True
        return False

# _UNUSED_TOKENIZER = models.load_tokenizer()
def get_dataset_fn(data_name):
    if data_name == 'human_eval':
        return human_eval.get_dataset
    if data_name == 'mbpp':
        return mbpp.get_dataset
    if data_name == 'ds1000':
        return ds1000.get_dataset
    if data_name == 'repo_eval':
        return repo_eval.get_dataset
    if data_name == 'evocodebench':
        return evocodebench.get_dataset
    if data_name == 'repoexec':
        return repo_exec.get_dataset
    if data_name == 'dev_eval':
        return dev_eval.get_dataset
    raise ValueError(f"Unknown dataset {data_name}")

def move_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu(v) for v in obj)
    else:
        return obj

@torch.no_grad()
def get_generations(model_name:str, args, seed=1, old_sequences=None, max_num_gen_once=args.max_num_gen_once, cache_dir='output'):
    device = args.device
    model, tokenizer = models.load_model_and_tokenizer(model_name, args.device, args.load_in_8bit)    
    utils.seed_everything(seed)
    print(model)
    model.eval()
    if 'chat' or 'instruct' in model_name.lower():
        instruction = True
    else:
        instruction = False
    dataset = get_dataset_fn(args.dataset)(tokenizer, language=args.language, instruction=instruction)
    if hasattr(dataset[0],'stopwords'):
        stop_words = dataset[0]['stopwords']
    else:
        stop_words = []
    if args.fraction_of_data_to_use < 1.0:
        dataset = dataset.train_test_split(test_size=(1 - args.fraction_of_data_to_use), seed=seed)['train']
    
    # dataset = dataset.select(range(len(dataset) - 300, len(dataset)))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    print('len dataset', len(dataloader))
    if old_sequences is None:
        old_sequences = []
    old_sequences = {_['task_id']: _ for _ in old_sequences}
    sequences = {}
    generation_sequences_output = []
    
    for batch_idx, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        task_id_path = str(batch['task_id'][0]).replace('/','_').replace('[','_').replace(']','_')
        out_dir_task_id = os.path.join(cache_dir, f"{task_id_path}.pkl")
        if batch['task_id'][0] in old_sequences:
            sequences.append(old_sequences[batch['task_id'][0]])
            continue
        if os.path.exists(os.path.join(cache_dir, f'llm_check_eig_prod_{task_id_path}.pkl')):
            print(f'Generated {task_id_path}!')
            continue # generated
        else:
            print(f'Processing {task_id_path} ...')
            # continue
        input_ids = batch['input_ids'].to(device)
        print(f"input_ids shape: {input_ids.shape}")
        if args.dataset not in passed_input_len_task  and (input_ids.shape[-1] > 1200 or input_ids.shape[-1] < 9):
            continue
        input_length = input_ids.shape[1]
        torch.cuda.empty_cache()
        
        # continue
        generations = []
        generations_decoded = []
        all_scores_softmax = []
        num_gens = args.num_generations_per_prompt
        all_token_hidden_states_layer_list = {}
        all_token_attentions_layer_list = {}
        lookback_ratio_list = {}
        llm_check_eig_prod_list = {}
        off_set = 0
        while num_gens > 0:
            dict_outputs =  model.generate(input_ids, attention_mask=batch['attention_mask'].to(device),
                                num_beams=1, max_new_tokens=args.max_new_tokens, 
                                num_return_sequences=min(max_num_gen_once, num_gens),
                                do_sample=True, 
                                top_p=args.top_p, 
                                top_k=args.top_k,
                                temperature=args.temperature, 
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.eos_token_id,
                                # output_hidden_states = True, 
                                output_scores=True,
                                output_attentions=True,
                                return_dict_in_generate=True, 
                            )
            print(dict_outputs.keys())
            # dict_outputs = move_to_cpu(dict_outputs)
            # print("Starting extracting...")
            # print(dict_outputs.keys())
            
            generation = dict_outputs.sequences[:, input_length:].cpu()
            for gen in generation:
                generations.append(gen)

            batch_scores = dict_outputs.scores
            ##### batch_scores : (num_tokens, num_seq, [vocab_size])
            batch_scores_softmax = [[] for _ in range(len(batch_scores[0]))]
            for ind1, logits in enumerate(batch_scores): 
                for ind2, seq_logits in enumerate(logits):
                    batch_scores_softmax[ind2].append(seq_logits.softmax(0)[generation[ind2][ind1]].cpu())
            
            all_scores_softmax.extend(batch_scores_softmax)
            
            layers_to_process = args.layers
            # hidden_states = dict_outputs.hidden_states
            ###### hidden_states : (num_tokens, num_layers, [num_seq, num_input_tokens/1, embedding_size])
            # for layer in layers_to_process:
            #     all_token_hidden_states_layer = {}
            #     for ind in range(hidden_states[1][-1].shape[0]):
            #         all_token_hidden_states_layer[ind + off_set] = []
            #         for hidden_state in hidden_states[1:]:
            #             all_token_hidden_states_layer[ind + off_set].append(hidden_state[layer][ind, -1, :].detach().cpu().float().numpy())

            #     if layer not in all_token_hidden_states_layer_list:
            #         all_token_hidden_states_layer_list[layer] = {}
            #     all_token_hidden_states_layer_list[layer].update(all_token_hidden_states_layer)
               
            attentions = dict_outputs.attentions
            ###### attentions: (num_tokens, num_layers, [num_seq, num_heads, seq_len, seq_len])

            context_length = attentions[0][0].shape[-1]
            new_token_length = len(attentions)
            num_layers = len(attentions[0])
            num_heads = attentions[0][0].shape[1]
            num_seq = attentions[0][0].shape[0]
            seq_len = attentions[-1][0].shape[-1]
            
            for ind in range(num_seq):
                attn_scores = {}
                for layer_num in range(num_layers):
                    attns = attentions[0][layer_num][ind, :, :, :].to(torch.float32).detach().cpu()
                    # print(attns.shape)
                    eigscore = 0.0
                    for attn_head_num in range(attns.shape[0]):  # iterating over number of attn heads
                        # attns[i][layer_num][j] is of size seq_len x seq_len
                        Sigma = attns[attn_head_num, :, :]
                        eigscore += torch.log(torch.diagonal(Sigma, 0)).mean()
                    attn_scores[layer_num] = eigscore
                llm_check_eig_prod_list[ind + off_set] = attn_scores
                
            # for ind in range(num_seq):
            #     lookback_ratio = torch.zeros((num_layers, num_heads, new_token_length))
            #     for i in range(new_token_length): # iterating over the new tokens length
            #         for l in range(num_layers):
            #             attn_on_context = attentions[i][l][ind, :, -1, :context_length].mean(-1)
            #             attn_on_new_tokens = attentions[i][l][ind, :, -1, context_length:].mean(-1)
            #             lookback_ratio[l, :, i] = attn_on_context / (attn_on_context + attn_on_new_tokens)
            #     lookback_ratio_list[ind + off_set] = lookback_ratio.detach().cpu().to(torch.float16)
            
            # # for layer in layers_to_process:
            # #     all_token_attention_layer = {}
            # #     # layer = layer - 1
            # #     for ind in range(num_seq):
            # #         attn = torch.zeros((num_heads, seq_len, new_token_length))
            # #         for i in range(new_token_length):
            # #             att_slice = attentions[i][layer - 1][ind, :, -1, :].detach().cpu().to(torch.float16)
            # #             actual_seq_len = att_slice.shape[1]
            # #             attn[:, :actual_seq_len, i] = att_slice
            # #             # attn[:, :, i] = attentions[i][layer - 1][ind, :, -1, :].detach().cpu().to(torch.float16)
            # #         all_token_attention_layer[ind + off_set] = attn
            # #     if layer not in all_token_attentions_layer_list:
            # #         all_token_attentions_layer_list[layer] = {}
            # #     all_token_attentions_layer_list[layer].update(all_token_attention_layer)
            
            del dict_outputs
            gc.collect()
            torch.cuda.empty_cache()
            layers = args.layers
            # del hidden_states
            gc.collect()
            torch.cuda.empty_cache()
            num_gens -= len(generation)
            off_set += len(generation)
        
        #### SAVE ALL INTERNAL STATES
        
        for gen_ids in generations:
            generations_decoded.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
        
        # for layer in layers_to_process:
        #     layer_embeddings = all_token_hidden_states_layer_list[layer]
        #     layer_embeddings_dict = dict(
        #             id=batch['task_id'][0],
        #             layer_embeddings = layer_embeddings,
        #         )
        #     layer_attention_dict = dict(
        #             id=batch['task_id'][0],
        #             layer_attention = all_token_attentions_layer_list[layer],
        #         )
        #     pd.to_pickle(layer_embeddings_dict, os.path.join(cache_dir, f'all_token_embedding_{task_id_path}_{layer}.pkl'))
        #     pd.to_pickle(layer_attention_dict, os.path.join(cache_dir, f'all_token_attention_{task_id_path}_{layer}.pkl'))
        
        # lookback_ratio_output = dict(
        #     id=batch['task_id'][0],
        #     lookback_ratio = lookback_ratio_list,
        # )    
        # pd.to_pickle(lookback_ratio_output, os.path.join(cache_dir, f'lookback_ratio_{task_id_path}.pkl'))
        
        llm_check_eig_prod_output = dict(
            id=batch['task_id'][0],
            llm_check_eig_prod = llm_check_eig_prod_list,
            softmax_scores=all_scores_softmax,
            generations = generations_decoded,
        )
        pd.to_pickle(llm_check_eig_prod_output, os.path.join(cache_dir, f'llm_check_eig_prod_{task_id_path}.pkl'))
        
        # generation_sequences_o
        # 
        # 
        # 
        # +
        # utput = dict(
        #         prompt=tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True),
        #         id=batch['task_id'][0],
        #         problem=batch['original_prompt'][0],
        #         generations=generations_decoded,
        #         generations_ids=generations,
        #         softmax_scores=all_scores_softmax,
        #     )
        # pd.to_pickle(generation_sequences_output, os.path.join(cache_dir, f'generation_sequences_output_{task_id_path}.pkl'))
        
        print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True))
        print("Problem:", batch['original_prompt'][0])
        print("AnswerGT:", batch['canonical_solution'][0])
        print("MostLikelyAns:", generations_decoded[0])
        
        print("Prompt:", tokenizer.decode(input_ids.cpu()[0], skip_special_tokens=True), file=logInfo)
        print("Problem:", batch['original_prompt'][0], file=logInfo)
        print("AnswerGT:", batch['canonical_solution'][0], file=logInfo)
        print("MostLikelyAns:", generations_decoded[0], file=logInfo)
        
        print("\n","\n","\n", file=logInfo)
        
        torch.cuda.empty_cache()
    
    return

def main(overwrite=False, continue_from=None, parallel:int=None):
    time_start = time.time()
    if continue_from:
        fname = os.path.basename(continue_from)
        args.__dict__ = utils.jload(continue_from.replace(fname, 'args'+fname.replace("_partial.pkl", ".json")))
        old_sequences = pd.read_pickle(continue_from)
        cache_dir = os.path.dirname(continue_from)
        run_id = int(os.path.basename(continue_from).replace("_partial.pkl", ""))
        model_name = args.model
    else:
        old_sequences = []
        model_name = args.model
        cache_dir = OUTPUT_DIR
        old_results = glob.glob(os.path.join(OUTPUT_DIR, '*.pkl'))
        old_results = [_ for _ in old_results if '_partial' not in _]
        run_id = len(old_results)
        with open(os.path.join(OUTPUT_DIR, f'args{run_id}.json'), 'w') as f:
            json.dump(args.__dict__, f)
    print(f'Generating {args.num_generations_per_prompt} generations per prompt for {model_name} on {args.dataset}...')
    print(f"Saving to {os.path.join(cache_dir, f'{run_id}.pkl')}")
    temp_dir = os.path.join(cache_dir,'temp2')
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    get_generations(model_name, args, seed=args.seed, old_sequences=old_sequences,cache_dir=temp_dir)
    print("Total time: ", time.time() - time_start)
    print("Total time: ", time.time() - time_start, file=logInfo)
    return

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    task_runner = main(parallel=args.nprocess)