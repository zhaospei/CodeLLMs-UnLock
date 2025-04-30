import json
import argparse
from collections import defaultdict
import glob
import pandas as pd
import os


def main(args):
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    label_df = pd.read_parquet(args.label_file)
    label_maping = dict()
    for _, row in label_df.iterrows():
        label_maping[row["completion_id"]] = row["label"]
    for file in glob.glob(f"{args.source_folder}/*.parquet"):
        dfsource = pd.read_parquet(file)
        labels = list()
        for _, row in dfsource.iterrows():
            labels.append(label_maping[row["completion_id"]])
        dfsource["label"] = labels
        layer = file.split("_")[-1].split(".")[0]
        layer = int(layer)
        out_dir = f"{args.outdir}/{layer}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        train_id = list()
        with open("train.txt") as f:
            train_id = [l.strip() for l in f.readlines()]
        train_df = dfsource[dfsource["task_id"].isin(train_id)]
        test_df = dfsource[~dfsource["task_id"].isin(train_id)]
        train_df.to_parquet(f"{out_dir}/train.parquet")
        test_df.to_parquet(f"{out_dir}/test.parquet")
        print(layer,train_df.shape, test_df.shape, test_df["label"].value_counts())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label_file",
        type=str,
    )
    parser.add_argument(
        "--source_folder",
        type=str,
    )
    parser.add_argument(
        "--outdir",
        type=str,
    )

    args = parser.parse_args()
    main(args)
