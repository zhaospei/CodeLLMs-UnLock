# /kaggle/input/lfclf-deepseekcoder-6-7b-instruct/standAlone_mbpp_test_task_id.txt
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
    dfs = pd.DataFrame(result)
    dfs.to_csv(f"result_{lang}_{field}.csv", index=False)


def get_magiccoder_path(layer):
    return f"mbpp/magiccoder/label/LFCLF_embedding_mbpp_ise-uiuc_Magicoder-S-DS-6.7B_{layer}_label.parquet"


def get_codellama_path(layer):
    return f"mbpp/codellama/LFCLF_embedding_mbpp_codellama_CodeLlama-7b-Instruct-hf_{layer}_label.parquet"


def get_deepseek_path(layer):
    return f"mbpp/deepseek67/LFCLF_embedding_mbpp_deepseek-ai_deepseek-coder-6.7b-instruct_{layer}.parquet"


def get_function_path(model):
    if model == "mc":
        get_path = get_magiccoder_path
    elif model == "cl":
        get_path = get_codellama_path
    elif model == "ds":
        get_path = get_deepseek_path
    return get_path


layers = [1, 4, 8, 12, 16, 20, 24, 28, 32]
from collections import defaultdict

tokens = [
    "first_token_embedding",
    "last_token_embedding",
    "first_token_code_embedding",
    "last_token_code_embedding",
]

import os


def get_data(model, par):
    get_path = get_function_path(model)

    if model == "ds":
        df = pd.read_parquet(
            "mbpp/deepseek67/LFCLF_2_embedding_mbpp_deepseek-ai_deepseek-coder-6.7b-instruct_32_label.parquet"
        )
    else:
        df = pd.read_parquet(get_path(32))
    embeddings = defaultdict(list)

    for layer in layers:
        tmp_emb = list()
        layer_df = pd.read_parquet(get_path(layer))
        layer_df = layer_df.reset_index()
        for idx, row in layer_df.iterrows():
            for f in tokens:
                embeddings[idx].append(row[f])
    embeddings = list(embeddings.values())
    df[f"emb"] = embeddings
    df["label"] = df["label"].apply(lambda x: 1 if x else 0)
    with open("mbpp/standAlone_mbpp_test_task_id.txt") as f:
        test_ids = [int(l.strip()) for l in f.readlines()]
    df["partition"] = df["task_id"].apply(
        lambda x: "test" if x in test_ids else "train"
    )
    # df.to_parquet(out_file)
    return df[df["partition"] == par]


models = ["mc", "cl", "ds"]
for model in models:
    train_models = [el for el in models if el != model]
    print(model)
    print(train_models)
    test_df = get_data(model, "test")
    root_train_df = get_data(model, "train")
    training_classify(root_train_df, test_df, "same_model", model)
    trains = list()
    for m in train_models:
        train_tmp = get_data(m, "train")
        # training_classify(train_tmp,test_df,f'{m}_train_{model}_test',model)
        trains.append(train_tmp)
    train_df = pd.concat(trains)
    training_classify(train_df, test_df, f"others_train_{model}_test", model)
