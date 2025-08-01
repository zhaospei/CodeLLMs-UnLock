import os
import time
import pickle
import pandas as pd
import multiprocessing

import numpy as np

import pickle


def normalize_list(arr_list, K):
    current_len = len(arr_list)
    if current_len > K:
        return arr_list[-K:]
    elif current_len < K:
        # Giả sử tất cả các phần tử đều có shape giống nhau
        pad_shape = arr_list[0].shape
        pad_array = np.zeros(pad_shape, dtype=arr_list[0].dtype)
        num_pad = K - current_len
        padding = [pad_array.copy() for _ in range(num_pad)]
        return padding + arr_list

    else:
        return arr_list


def get_embedding(task_id, layer, completion_id, max_length=360):
    # print("start")
    try:
        embedding_cache = dict()
        cache_file = f"{max_length}/{task_id}_{layer}.pkl"
        print(cache_file)
        completion_id = completion_id.split("_")[-1]
        key = f"{task_id}_{layer}_{completion_id}"
        if os.path.exists(cache_file):
            return
            embedding_cache = pickle.load(open(cache_file, "rb"))
            # print("end cache")
            return embedding_cache[key]
        with open(
            f"/root/security/WCODELLM/output/codegemma_mbpp_full/all_token_embedding_tensor({task_id})_{layer}.pkl",
            "rb",
        ) as f:
            data = pickle.load(f)
        for k, v in data["layer_embeddings"].items():
            tmp_key = f"{task_id}_{layer}_{k}"
            tmp_vector = normalize_list(v, max_length)
            embedding_cache[tmp_key] = tmp_vector
        with open(cache_file, "wb") as ff:
            pickle.dump(embedding_cache, ff)
    except Exception as e:
        print(e)
    # print("end")
    # return embedding_cache[key]
    return


# get_embedding("11", 1, "11_0")
layers = [1, 4, 8, 12, 16, 20, 24, 28]


def main(args):
    NUM_CORES = 20  # Số core muốn dùng
    NUM_TASKS = 40  # Số task cần xử lý
    df = pd.read_parquet(
        "/root/security/WCODELLM/intrinsic/output2/LFCLF_2_embedding_mbpp_google_codegemma-7b-it_1_compliable_label_output_error.parquet"
    )
    # return
    trace = list()
    pool = multiprocessing.Pool(processes=20)

    for i, row in df.iterrows():
        for l in layers:
            tmp = pool.apply_async(
                get_embedding,
                args=(row["task_id"], l, row["completion_id"], args.MAX_TOKEN),
            )
            trace.append(tmp)
    pool.close()
    pool.join()


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--MAX_TOKEN",
        type=int,
    )
    args = parser.parse_args()
    main(args)
