import pandas as pd
import json
import argparse
from collections import defaultdict
import glob
import pandas as pd
import os


def main(args):
    # file = "LFCLF_embedding_security_ise-uiuc_Magicoder-S-DS-6.7B_1.parquet"
    file = args.file
    df = pd.read_parquet(file)
    # out_dir = f"source/{file}"
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for _, row in df.iterrows():
        out_file = f'{row["completion_id"]}.py'
        out_path = os.path.join(out_dir, out_file)
        content = row["clean_decode"]
        if content.startswith("\t") or content.startswith(" "):
            content = row["prompt"] + "\n" + content
        with open(out_path, "w+") as f:
            f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
    )
    parser.add_argument(
        "--out_dir",
        type=str,
    )
    args = parser.parse_args()
    main(args)
