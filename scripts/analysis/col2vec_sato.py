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
    mcol_model.load_state_dict(torch.load(sato_mcol_model_path,
                                             map_location=device))
    mcol_model.eval()


    scol_model = BertForSequenceClassification.from_pretrained(
        shortcut_name,
        num_labels=78,
        output_attentions=True,
        output_hidden_states=True,
    ).to(device)
    scol_model.load_state_dict(torch.load(sato_scol_model_path,
                                          map_location=device))
    scol_model.eval()

    mcol_dataset = SatoCVTablewiseDataset(0, split="test", tokenizer=tokenizer)
    scol_dataset = SatoCVColwiseDataset(0, split="test", tokenizer=tokenizer)

    mcol_dataloader = DataLoader(mcol_dataset,
                                 batch_size=batch_size,
                                 collate_fn=collate_fn)
    scol_dataloader = DataLoader(scol_dataset,
                                 batch_size=batch_size,
                                 collate_fn=collate_fn)

    ## Single column
    """
    scol_emb_list = []
    #for i, s_data in tqdm(enumerate(scol_dataset)):
    for i in range(len(scol_dataset)):
        s_data = scol_dataset[i]
        outputs = scol_model(s_data["data"].unsqueeze(0).T.to(device),
                             output_attentions=True,
                             output_hidden_states=True)
        emb = outputs.hidden_states[-1][0].squeeze(0).detach().cpu().numpy()
        scol_emb_list.append(emb)

    scol_df = scol_dataset.df.copy()
    scol_df = scol_df.drop(columns=["data_tensor", "label_tensor"])
    scol_df["emb"] = scol_emb_list

    with open("data/sato_scol_emb_df.pickle", "wb") as fout:
        pickle.dump(scol_df, fout)
    """

    ## Multi-column model
    mcol_data_list = []
    #for i, m_data in tqdm(enumerate(mcol_dataset)):
    for i in range(len(mcol_dataset)):
        m_data = mcol_dataset[i]
        outputs = mcol_model.bert.encoder(mcol_model.bert.embeddings(
            m_data["data"].unsqueeze(0).T.to(device)),
            output_attentions=True,
            output_hidden_states=True)
        last_hidden_states = outputs.last_hidden_state
        cls_indexes = torch.nonzero(
            m_data["data"] == tokenizer.cls_token_id).squeeze(1).detach().cpu().numpy()
        print(cls_indexes)
        for cls_id, cls_index in zip(m_data["label"].detach().cpu().numpy(), cls_indexes):
            emb = last_hidden_states[cls_index].squeeze(0).detach().cpu().numpy()  # 768
            mcol_data_list.append([
                mcol_dataset.table_df.iloc[i]["table_id"],
                cls_id,
                sato_coltypes[cls_id],
                emb
            ])
    mcol_df = pd.DataFrame(mcol_data_list,
                           columns=["table_id",
                                    "class_id",
                                    "class",
                                    "emb"])
    with open("data/sato_mcol_emb_df.pickle", "wb") as fout:
        pickle.dump(mcol_df, fout)



