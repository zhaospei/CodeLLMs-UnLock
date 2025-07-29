with open('../openai/test_ids/mbpp_eval_test_ids.txt') as f:
    test_ids = [int(l.strip())   for l in f.readlines()]

import json
result = list()
with open('mbpp.jsonl' ) as f:
    for line in f.readlines():
        data = json.loads(line.strip())
        if data['task_id'] in test_ids:
            result.append(data) 
with open('mbpp_test.jsonl', 'w') as f:
    for line in result:
        f.write(json.dumps(line) + '\n')
        
from openai import OpenAI
client = OpenAI(
    api_key = 'TODO'
)

PROMPT = """
Paraphrase the following code generation request to make it clearer and easier to understand, while preserving the original meaning.
Ensure the rephrased version remains concise.
Here is the input request:
%s
"""

def get_new_req(req):
    prompt =  PROMPT % req
    res_openai = client.chat.completions.create(
    model = 'gpt-4o',
    messages = [
        {
            'role': 'user',
            'content': prompt
        }
    ]
    )
    return res_openai.choices[0].message.content

new_res = list()
from tqdm import tqdm
for el in tqdm(result):
    # print(el)
    text = el['text']
    el['text'] = get_new_req(text)
    el['old_text'] = text
    new_res.append(el)
with open('mbpp_test_parapharse.jsonl', 'w') as f:
    for line in new_res:
        f.write(json.dumps(line) + '\n')