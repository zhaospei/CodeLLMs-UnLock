import pandas as pd
import torch
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random


class BinaryClassifier(nn.Module):
    def __init__(self, embedding_dim):
        super(BinaryClassifier, self).__init__()

        self.fc_select = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, 1)
        )
        self.soft_max = nn.Softmax(dim=-1)

        self.fc_classify = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scores = self.fc_select(x)
        scores = F.gumbel_softmax(scores, tau=1.0, dim=-2, hard=False)
        selected_vectors = x * scores
        selected_vectors = selected_vectors.sum(dim=1)
        output = self.fc_classify(selected_vectors)
        return output, scores


from sklearn.metrics import accuracy_score


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(embeddings)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 and False:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}"
            )


# Testing function
def test_model(model, test_loader):
    model.eval()
    result = list()
    predictions, true_labels = [], []
    with torch.no_grad():
        for embeddings, labels in test_loader:
            outputs, scores = model(embeddings)
            result.append([embeddings, scores])
            outputs = outputs.squeeze()
            preds = (
                outputs > 0.5
            ).float()  # Convert probabilities to binary predictions
            predictions.extend(preds.tolist())
            true_labels.extend(labels.tolist())

    accuracy = accuracy_score(true_labels, predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return predictions, true_labels, result


def get_data_for_training(data, field):
    embeddings = list()
    labels = list()
    for idx, row in data.iterrows():
        labels.append(row["label"])
        embeddings.append(row["emb"])
    print(embeddings[0])
    embeddings = torch.tensor(np.array(embeddings), device=device).to(torch.float)
    labels = torch.tensor(labels, device=device)
    print(embeddings.shape, labels.shape)
    return embeddings, labels


from sklearn.metrics import (
    classification_report,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
)
import pickle


def training_classify(train_df, test_df, field, lang="CPP"):
    X_train, y_train = get_data_for_training(train_df, field)
    X_test, y_test = get_data_for_training(test_df, field)
    embedding_dim = X_train.shape[-1]
    num_epochs = 50
    batch_size = 32
    learning_rate = 0.001

    # Split data into train and test sets

    # Data loaders
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    test_data = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    results = list()
    precs = list()
    recalls = list()
    f1s = list()
    accs = list()
    ps, refs = list(), list()
    vectors = list()
    for i in range(5):
        model = BinaryClassifier(embedding_dim)
        model = model.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Training the model
        train_model(model, train_loader, criterion, optimizer, num_epochs)

        # Testing the model
        ps, refs, vectors = test_model(model, test_loader)
        tmp_r = recall_score(refs, ps, average="weighted")
        tmp_p = precision_score(refs, ps, average="weighted")
        tmp_f = f1_score(refs, ps, average="weighted")
        tmp_a = accuracy_score(refs, ps)
        precs.append(tmp_p)
        recalls.append(tmp_r)
        f1s.append(tmp_f)
        accs.append(tmp_a)
        results.append(classification_report(refs, ps))
    with open(f"{lang}_vectors.pkl", "wb") as ff:
        pickle.dump(vectors, ff)
    for el in results:
        print(el)

    print("recall", recalls)
    print("precision", precs)
    print("f1", f1s)
    print("acc", accs)
    max_index = accs.index(max(accs))
    print(
        f"mean {lang} {field}: {sum(accs)/5}\t{sum(precs)/5}\t{sum(recalls)/5}\t{sum(f1s)/5}"
    )
    print(
        f"max {lang} {field}: {accs[max_index]}\t{precs[max_index]}\t{recalls[max_index]}\t{f1s[max_index]}\t"
    )
    result = list()
    test_df = test_df.reset_index()
    for idx, row in test_df.iterrows():
        result.append(
            {
                "ref": refs[idx],
                "predict": ps[idx],
                "id": row["task_id"],
                "code": row["extracted_code"],
                "completion_id": row["completion_id"],
            }
        )
    # dfs = pd.DataFrame(result)
    # dfs.to_csv(f"result_{lang}_{field}.csv", index=False)


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


# embedding_cache = dict()


def get_embedding(task_id, layer, completion_id, max_length=360):
    # print("start")
    embedding_cache = dict()
    cache_file = f"token_sampling/{max_length}/{task_id}_{layer}.pkl"

    completion_id = completion_id.split("_")[-1]
    key = f"{task_id}_{layer}_{completion_id}"
    if os.path.exists(cache_file):
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
    # print("end")
    return embedding_cache[key]


# /kaggle/input/lfclf-deepseekcoder-6-7b-instruct/standAlone_mbpp_test_task_id.txt
from collections import defaultdict

tokens = [
    "first_token_embedding",
    "last_token_embedding",
    "first_token_code_embedding",
    "last_token_code_embedding",
]

import os

import pandas as pd

layers = [1, 4, 8, 12, 16, 20, 24, 28]

def get_data(model, par, MAX_TOKEN):

    df = pd.read_parquet(
        "output2/LFCLF_2_embedding_mbpp_google_codegemma-7b-it_1_compliable_label_output_error.parquet"
    )
    file_name = f"proceed_{MAX_TOKEN}.parquet"
    if os.path.exists(file_name):
        df = pd.read_parquet(file_name)
        return df[df["partition"] == "train"], df[df["partition"] == "test"]
    embeddings = defaultdict(list)
    for layer in tqdm(layers):
        print("process layer:", layer)
        tmp_emb = list()
        for idx, row in df.iterrows():
            emb_tmp = get_embedding(
                row["task_id"], layer, row["completion_id"], MAX_TOKEN
            )
            embeddings[idx].extend(emb_tmp)
    embeddings = list(embeddings.values())
    df[f"emb"] = embeddings
    df["label"] = df["label"].apply(lambda x: 1 if x else 0)
    with open("mbpp/standAlone_mbpp_test_task_id.txt") as f:
        test_ids = [int(l.strip()) for l in f.readlines()]
    df["partition"] = df["task_id"].apply(
        lambda x: "test" if x in test_ids else "train"
    )
    df.to_parquet(f"proceed_{MAX_TOKEN}.parquet")
    return df[df["partition"] == "train"], df[df["partition"] == "test"]


def main(args):
    models = ["cg"]
    for model in models:
        train_models = [el for el in models if el != model]
        root_train_df, test_df = get_data(model, "test", args.MAX_TOKEN)
        training_classify(root_train_df, test_df, "same_model", model)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--MAX_TOKEN",
        type=int,
    )
    args = parser.parse_args()
    main(args)
