
import argparse
import glob 
import os
import time
import datetime
from pathlib import Path
from openai import OpenAI


def run(args):
    with open('openai/.key') as f:
        key = f.read().strip()
    client = OpenAI(
        api_key = key
    )
    print(f"Start call api: {args.input_dir}")
    input_temp = os.path.join(args.input_dir,'*.jsonl')
    for file in glob.glob(input_temp):
        file_name = str(Path(file).name)
        if file_name.startswith('output'):
            continue
        
        if 'mini_batch' not in file_name:
            continue
        # print(f'process: {file}')
        output_file = os.path.join(args.output_dir,'output_'+str(Path(file).name))
        if os.path.exists(output_file):
            continue
        print('process file:',file)
        # continue
        file_openai = client.files.create(
            file=Path(file),
            purpose="batch",
        )
        file_id = file_openai.id 
        tmp_batch = client.batches.create(
            input_file_id= file_id,
            endpoint=args.endpoint,
            completion_window="24h",
            metadata={
            "description": "automated batch"
            }
        )
        batch_id = tmp_batch.id
        print('create batch',batch_id)
        while True:
            time.sleep(int(args.sleep_time)) #check status batch every 60s
            try:
                batch = client.batches.retrieve(batch_id)
                if batch.status == 'completed':
                    if batch.output_file_id:
                        client.files.content(batch.output_file_id).write_to_file(output_file)
                        print('='*33)
                        print(f'done: {file}')
                        print(f'output: {output_file}')
                    elif batch.error_file_id:
                        client.files.content(batch.output_file_id).write_to_file(output_file+'.error')
                        print('='*33)
                        print(f'done: {file}')
                        print(f'error file: {output_file}')
                    break
                elif batch.status == 'failed':
                    print(f"Faild to call api: {file}")
                    print(batch.errors.data[0].message)
                    break
                else:
                    print(f'{datetime.datetime.now()}: Attempt check status batch - {batch.status}')
                    print(batch.request_counts)
            except Exception as e:
                print(e)
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir")
    parser.add_argument("--output_dir")
    parser.add_argument('--sleep_time',default=60)
    parser.add_argument("--endpoint",default="/v1/chat/completions")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()