import pandas as pd
import argparse
import os
import glob
import json
def main(args):
    models = ['ds13','ds','clm','mgcd']
    models =['clm13']
    datasets = ['human_eval','mbpp','dev_eval']
    # datasets = ['mbpp']
    model_gpt_file = ['deepseek13','deepseek6.7','codellama7','magiccoder6.7']
    model_gpt_file = ['codellama13']
    root_out_file = 'openai/batch_clm13_o4'
    for i in range(len(models)):
        for dataset in datasets:
            outfile_openai_pattern = f'{root_out_file}/output_batch_{model_gpt_file[i]}_{dataset}_*.jsonl'
            source_file = f'openai/data/test_{dataset}_{models[i]}.parquet'
            out_file = f'openai/data/gpt_test_{dataset}_{models[i]}_gpto4mn.parquet'
            print('outfile_openai_pattern: ',outfile_openai_pattern)
            print('source file:',source_file)
            print('outfile:', out_file)
            df = pd.read_parquet(source_file)
            openai_res = dict()
            for file in glob.glob(outfile_openai_pattern):
                with open(file) as f:
                    data = [json.loads(l.strip()) for l in f.readlines()]
                    for tmp in data:
                        index = tmp['custom_id']
                        result = tmp['response']['body']['choices'][0]['message']['content'].strip()
                        if len(result) <= 0 :
                            result = '0'
                        else:
                            result = result
                        openai_res[index] = result
            print('len openai:',len(openai_res))
            print('len df: ',df.shape)
            results = list()
            for _,row in df.iterrows():
                index = row['completion_id']
                if index not in openai_res:
                    row['gpt_result'] = ''
                    print(index)
                else:
                    row['gpt_result'] = openai_res[index].strip()
                results.append(row)
            dfs = pd.DataFrame(results)
            print('save to file:',out_file)
            dfs.to_parquet(out_file)
            
            print('_'*33)
    

if __name__ == "__main__":
    # Load the DataFrame from a Parquet file
    args = argparse.ArgumentParser()
    args.add_argument("--file", type=str,default='')
    args.add_argument("--output_openai_file", type=str,default='')
    args.add_argument("--out_file", type=str,default='')
    args = args.parse_args()
    main(args)