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
    from matplotlib import pyplot as plt
    from sklearn.manifold import TSNE
    import seaborn as sns

    with open("data/sato_scol_emb_df.pickle", "rb") as fin:
        scol_df = pickle.load(fin)
    with open("data/sato_mcol_emb_df.pickle", "rb") as fin:
        mcol_df = pickle.load(fin)

    """
    selected_list = ["currency",
                     "classification",
                     "species",
                     "birthPlace",
                     "capacity",
                     "ranking",
                     "birthDate",
                     "command",
                     "education",
                     "continent"]
    """
    selected_list = ["name",
                     "description"
                     "team",
                     "age",
                     "type",
                     "city",
                     "year"
                     "location",
                     "rank",
                     "status"]

    for name, df in [("mcol", mcol_df),
                     ("scol", scol_df)]:
        if os.path.exists("data/viznet_doduo_tsne_{}.npy".format(name)):
            X2d = np.load("data/viznet_doduo_tsne_{}.npy".format(name))
        else:
            tsne = TSNE(n_components=2)
            X = np.array(df["emb"].tolist())
            X2d = tsne.fit_transform(X)
            np.save("data/viznet_doduo_tsne_{}.npy".format(name), X2d)

        df["emb_2d"] = X2d.tolist()

        ## Filter
        filtered_df = df[df["class"].isin(selected_list)]
        X2d = np.array(filtered_df["emb_2d"].tolist())

        size = [1 for _ in range(len(filtered_df))]
        sns.scatterplot(X2d[:, 0], X2d[:, 1], size=size, hue=filtered_df["class"].tolist())
        plt.tight_layout()
        plt.savefig("fig/col2vec_viznet_doduo_{}_tsne_2d_10classes.pdf".format(name))
        plt.close()
