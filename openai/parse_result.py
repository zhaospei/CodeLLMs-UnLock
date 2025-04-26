import pandas as pd
import argparse
import os


def get_prompt_dataset(dataset):
    result = dict()
    if dataset == "hummaneval":
        result = []
    elif dataset == "mbpp":
        result = []  
    elif dataset == "deveval":
        result = [] 
    return result

def main(args):
    df = pd.read_parquet(args.file)
    prompt = get_prompt_dataset(args.dataset)
    for i, row in df.iterrows():
        pass
    pass

if __name__ == "__main__":
    # Load the DataFrame from a Parquet file
    args = argparse.ArgumentParser()
    args.add_argument("--file", type=str)
    args.add_argument("--output", type=str)
    args.add_argument("--dataset", type=str )
    args.add_argument("--model", type=str)
    args = args.parse_args()
    main(args)