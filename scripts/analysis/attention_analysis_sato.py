import argparse
from collections import Counter, defaultdict
import math
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification

from dataset import (
    SatoTablewiseDataset,
    collate_fn,
    TURLColTypeColwiseDataset,
    TURLColTypeTablewiseDataset,
    TURLRelExtColwiseDataset,
    TURLRelExtTablewiseDataset,
    SatoCVColwiseDataset,
    SatoCVTablewiseDataset,
)

from model import BertForMultiOutputClassification, BertMultiPairPooler

import seaborn as sns
from matplotlib import pyplot as plt


sato_coltypes = ["address", "affiliate", "affiliation", "age", "album", "area",
                 "artist", "birthDate", "birthPlace", "brand", "capacity", "category",
                 "city", "class", "classification", "club", "code", "collection", "command",
                 "company", "component", "continent", "country", "county", "creator", "credit",
                 "currency", "day", "depth", "description", "director", "duration", "education",
                 "elevation", "family", "fileSize", "format", "gender", "genre", "grades", "isbn",
                 "industry", "jockey", "language", "location", "manufacturer", "name", "nationality",
                 "notes", "operator", "order", "organisation", "origin", "owner", "person", "plays",
                 "position", "product", "publisher", "range", "rank", "ranking", "region", "religion",
                 "requirement", "result", "sales", "service", "sex", "species", "state", "status",
                 "symbol", "team", "teamName", "type", "weight", "year"]


## Except for the following 3 classes, the column type name is a single token
# 2 birthDate
# 2 fileSize
# 2 teamName
"""
cnt = Counter()
for coltype in sato_coltypes:
    num_subtokens = len(tokenizer.encode(coltype, add_special_tokens=False))
    cnt[num_subtokens] += 1
    if num_subtokens > 1:
        print(num_subtokens, coltype)
"""

## TURL coltype
# Counter({3: 133, 1: 80, 5: 29, 9: 1, 4: 5, 7: 4, 6: 2, 2: 1})

## TURL reltype

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpu",
        action="store_true"
    )
    args = parser.parse_args()

    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    shortcut_name = "bert-base-uncased"
    batch_size = 1
    max_length = 32

    ## Load models
    sato_mcol_model_path = "model/sato0_mosato_bert_bert-base-uncased-bs16-ml-32__sato0-1.00_best_macro_f1.pt"
    sato_scol_model_path = "model/sato0_single_bert_bert-base-uncased-bs16-ml-32__sato0-1.00_best_macro_f1.pt"

    ## Load mlb
    tokenizer = BertTokenizer.from_pretrained(shortcut_name)
    mcol_model = BertForMultiOutputClassification.from_pretrained(
        shortcut_name,
        num_labels=78,
        output_attentions=True,
        output_hidden_states=True,
    ).to(device)
    #mcol_model.load_state_dict(torch.load(sato_mcol_model_path,
    #                                         map_location=device))
    mcol_model.eval()

    scol_model = BertForSequenceClassification.from_pretrained(
        shortcut_name,
        num_labels=78,
        output_attentions=True,
        output_hidden_states=True,
    ).to(device)
    #scol_model.load_state_dict(torch.load(sato_scol_model_path,
    #                                     map_location=device))
    scol_model.eval()

    mcol_dataset = SatoCVTablewiseDataset(0, split="test",
                                          tokenizer=tokenizer,
                                          max_length=max_length,
                                          device=device)
    #scol_dataset = SatoCVColwiseDataset(0, split="test", tokenizer=tokenizer)

    ## Multi-column model
    mcol_data_list = []

    cooc_matrix_list = []
    att_matrix_list = []
    #for i, m_data in tqdm(enumerate(mcol_dataset)):
    for i in tqdm(range(len(mcol_dataset))):
        m_data = mcol_dataset[i]
        outputs = mcol_model.bert.encoder(mcol_model.bert.embeddings(
            m_data["data"].unsqueeze(0).to(device)),
            output_attentions=True,
            output_hidden_states=True)

        cls_indexes = torch.nonzero(
            m_data["data"] == tokenizer.cls_token_id).squeeze(1).detach().cpu().numpy()
        if len(cls_indexes) < 2:
            continue

        last_attentions = outputs.attentions[-1]
        A = last_attentions.squeeze(0).mean(0)
        Acls = torch.index_select(torch.index_select(A, 0, torch.LongTensor(cls_indexes).to(device)), 1,
                                  torch.LongTensor(cls_indexes).to(device)).detach().cpu().numpy()
        Acls = Acls / Acls.sum(axis=1, keepdims=1) # Normalize

        # print(cls_indexes)
        att_matrix = np.empty((78, 78))
        att_matrix[:] = np.nan

        cooc_matrix = np.empty((78, 78))
        cooc_matrix[:] = np.nan

        for j, cls_id1 in enumerate(m_data["label"].detach().cpu().numpy()):
            for k, cls_id2 in enumerate(m_data["label"].detach().cpu().numpy()):
                att_matrix[cls_id1][cls_id2] = Acls[j][k]
                cooc_matrix[cls_id1][cls_id2] = 1.0 / len(m_data["label"])

        att_matrix_list.append(att_matrix)
        cooc_matrix_list.append(cooc_matrix)
        """
        for cls_id, cls_index in zip(m_data["label"].detach().cpu().numpy(), cls_indexes):
            mcol_data_list.append([
                mcol_dataset.table_df.iloc[i]["table_id"],
                cls_id,
                sato_coltypes[cls_id],
                emb
            ])
        """

    att_matrix_mean = np.nanmean(np.array(att_matrix_list), axis=0)
    cooc_matrix_mean = np.nanmean(np.array(cooc_matrix_list), axis=0)
    #np.save("data/att_matrix_msato_cv0_attention_mcol.npy", att_matrix_mean)
    #np.save("data/cooc_matrix_msato_cv0_attention_mcol.npy", cooc_matrix_mean)

    att_df = pd.DataFrame(att_matrix_mean, columns=sato_coltypes)
    att_df.index = sato_coltypes

    cooc_df = pd.DataFrame(cooc_matrix_mean, columns=sato_coltypes)
    cooc_df.index = sato_coltypes

    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(att_df, cmap="Reds", ax=ax)
    plt.close()

    fig, ax = plt.subplots(figsize=(16, 16))
    sns.heatmap(cooc_df, cmap="Reds", ax=ax)
    plt.close()


    fig, ax = plt.subplots(figsize=(24, 24))
    sns.heatmap(att_df.div(cooc_df) - 1, cmap="coolwarm", ax=ax)
    plt.tight_layout()
    plt.savefig("fig/att_cooc_msato_cv0_mcol_doduo.pdf")
    plt.close()

    ratio_df = (att_df.div(cooc_df) - 1)
    #filtered_ratio_df = ratio_df[(ratio_df.isnull().sum(axis=0) < 68)]
    filtered_ratio_df = ratio_df[(ratio_df.isnull().sum(axis=0) < 73)]
    filtered_ratio_df = filtered_ratio_df[filtered_ratio_df.index.tolist()]

    sorted_index = filtered_ratio_df.mean().sort_values().index
    remove_list = ["ranking",
                   "creator",
                   "plays",
                   "sex",
                   "brand",
                   "year",
                   "symbol",
                   "publisher",
                   "album",
                   "genre",
                   "notes",
                   "artist",
                   "depth",
                   "order",
                   "rank",
                   "weight",
                   "code",
                   "format",
                   "position",
                   "status",
                   "category",
                   "result",
                   "component",
                   "day",
                   "club"]

    sorted_index = list(filter(lambda x: x not in remove_list, sorted_index))

    fig, ax = plt.subplots(figsize=(16, 12))
    #sns.set(font_scale=3)

    sns.heatmap(filtered_ratio_df.loc[sorted_index, sorted_index],
                cmap="coolwarm",
                vmin=-0.15,
                vmax=0.15,
                center=0,
                ax=ax)

    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)

    plt.tight_layout()
    plt.savefig("fig/att_cooc_msato_cv0_mcol_doduo_filtered.pdf")
    plt.close()






