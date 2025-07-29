from transformers import RobertaTokenizer
import json
MAX = 150000
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
file_postfix = 'mini_batch'
root = "batch_clm13_o4/"
source_folder = root
import glob
for file_process in glob.glob(f'{source_folder}/*.jsonl'):
    file_name = file_process.split("/")[-1][:-6]
    print(file_name)
    index = 0
    count = 0
    result = list()
    with open(file_process) as f:
        for line in f.readlines():
            obj = json.loads(line.strip())
            content = obj['body']['messages'][0]['content']
            tokens = tokenizer.tokenize(content)
            if count + len(tokens) >MAX:
                with open(f"{root}{file_name}_{file_postfix}_{index}.jsonl",'w+') as ff:
                    for tmp in result:
                        ff.writelines(json.dumps(tmp) + '\n')
                print('1',len(result))
                count = len(tokens)
                result = [obj]
                index += 1
            else:
                result.append(obj)
                count += len(tokens)
    with open(f"{root}{file_name}_{file_postfix}_{index}.jsonl",'w+') as ff:
        for tmp in result:
            ff.writelines(json.dumps(tmp) + '\n')
    print('final',len(result))