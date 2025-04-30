import json
import argparse
from collections import defaultdict
import glob
import pandas as pd

def main(args):
    files_sec = list()
    results = list()
    with open(args.res_file) as f:
        result_sec = json.load(f)
        files_sec = result_sec.keys()
    for file in glob.glob(f'{args.source_folder}/*.py'):
        content = ''
        with open(file) as fff:
            content =fff.read()
        file_name = file.split('/')[-1]
        label = 1 if file_name in files_sec else 0
        error = json.dumps(result_sec[file_name]) if label else ''
        task_id = file_name[:-5]
        completion_id = file_name[:-3]
        results.append({"completion_id":completion_id,'task_id':task_id,'extracted_code':content,'label':label,'detail':error})
    df = pd.DataFrame(results)
    df.to_parquet(args.outfile)
    print(df.shape)
    print(df.head(1))
    print('_'*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--res_file",
        type=str,
    )
    parser.add_argument(
        "--source_folder",
        type=str,
    )
    parser.add_argument(
        "--outfile",
        type=str,
    )

    args = parser.parse_args()
    main(args)
