import argparse
from collections import defaultdict
import json
import math
import os
import sys
from time import time
from functools import reduce
import operator
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Dataset

import transformers
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, BertConfig

from doduo.dataset import collate_fn
from doduo.model import BertForMultiOutputClassification, BertMultiPairPooler
from doduo.util import parse_tagname, f1_score_multilabel

sato_coltypes = [
    "address", "affiliate", "affiliation", "age", "album", "area", "artist",
    "birthDate", "birthPlace", "brand", "capacity", "category", "city",
    "class", "classification", "club", "code", "collection", "command",
    "company", "component", "continent", "country", "county", "creator",
    "credit", "currency", "day", "depth", "description", "director",
    "duration", "education", "elevation", "family", "fileSize", "format",
    "gender", "genre", "grades", "isbn", "industry", "jockey", "language",
    "location", "manufacturer", "name", "nationality", "notes", "operator",
    "order", "organisation", "origin", "owner", "person", "plays", "position",
    "product", "publisher", "range", "rank", "ranking", "region", "religion",
    "requirement", "result", "sales", "service", "sex", "species", "state",
    "status", "symbol", "team", "teamName", "type", "weight", "year"
]


class AnnotatedDataFrame:

    def __init__(self, df):
        self.df = df


class DFColTypeTablewiseDataset(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        data_list = []
        for i in range(len(df.columns)):
            data_list.append([
                1,  # Dummy table ID (fixed)
                0,  # Dummy label ID (fixed)
                " ".join([str(x) for x in df.iloc[:, i].dropna().tolist()])
            ])
        self.df = pd.DataFrame(data_list,
                               columns=["table_id", "label_ids", "data"])

        # For learning curve
        num_tables = len(self.df.groupby("table_id"))

        data_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            token_ids_list = group_df["data"].apply(lambda x: tokenizer.encode(
                x, add_special_tokens=True, max_length=max_length + 2)).tolist(
                )
            token_ids = torch.LongTensor(reduce(operator.add,
                                                token_ids_list)).to(device)
            cls_index_list = [0] + np.cumsum(
                np.array([len(x) for x in token_ids_list])).tolist()[:-1]
            for cls_index in cls_index_list:
                assert token_ids[
                    cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
            cls_indexes = torch.LongTensor(cls_index_list).to(device)
            class_ids = torch.LongTensor(
                group_df["label_ids"].tolist()).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }


class DFColTypeColwiseDataset(Dataset):

    def __init__(self,
                 df: pd.DataFrame,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 device: torch.device = None):
        if device is None:
            device = torch.device("cpu")

        data_list = []
        for i in range(len(df.columns)):
            data_list.append([
                1,  # Dummy table ID (fixed)
                0,  # Dummy label ID (fixed)
                " ".join([str(x) for x in df.iloc[:, i].dropna().tolist()])
            ])
        self.df = pd.DataFrame(data_list,
                               columns=["table_id", "label_ids", "data"])
        self.df["data_tensor"] = self.df["data"].apply(
            lambda x: torch.LongTensor(
                tokenizer.encode(
                    x, add_special_tokens=True, max_length=max_length + 2)).to(
                        device)).tolist()

        self.df["label_tensor"] = self.df["label_ids"].apply(
            lambda x: torch.LongTensor([x]).to(device))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"]
        }


class Doduo:

    def __init__(self, args=None, basedir="./"):
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")

        if args is None:
            args = argparse.Namespace

        self.args = args
        self.args.colpair = True
        self.args.shortcut_name = "bert-base-uncased"
        self.args.batch_size = 16

        ## Load models
        self.tokenizer = BertTokenizer.from_pretrained(self.args.shortcut_name)

        if self.args.model == "viznet":
            coltype_model_path = os.path.join(
                basedir,
                "model/sato0_mosato_bert_bert-base-uncased-bs16-ml-32__sato0-1.00_best_micro_f1.pt"
            )
            coltype_num_labels = 78
        elif self.args.model == "wikitable":
            coltype_model_path = os.path.join(
                basedir,
                "model/turlturl-re-colpair_mosato_bert_bert-base-uncased-bs16-ml-16__turl-1.00_turl-re-1.00=turl_best_micro_f1.pt"
            )
            colrel_model_path = os.path.join(
                basedir,
                "model/turlturl-re-colpair_mosato_bert_bert-base-uncased-bs16-ml-16__turl-1.00_turl-re-1.00=turl-re_best_micro_f1.pt"
            )
            coltype_num_labels = 255
            colrel_num_labels = 121

            ## Load mlb
            with open(os.path.join(basedir, "data/turl_coltype_mlb.pickle"),
                      "rb") as fin:
                self.coltype_mlb = pickle.load(fin)
            with open(os.path.join(basedir, "data/turl_colrel_mlb.pickle"),
                      "rb") as fin:
                self.colrel_mlb = pickle.load(fin)
        else:
            raise ValueError("Invalid args.model: {}".format(args.model))

        # ===== Not necessary?
        coltype_config = BertConfig.from_pretrained(
            self.args.shortcut_name,
            num_labels=coltype_num_labels,
            output_attentions=True,
            output_hidden_states=True)
        # =====

        self.coltype_model = BertForMultiOutputClassification.from_pretrained(
            self.args.shortcut_name,
            num_labels=coltype_num_labels,
            output_attentions=True,
            output_hidden_states=True,
        ).to(self.device)
        self.coltype_model.load_state_dict(
            torch.load(coltype_model_path, map_location=self.device))
        self.coltype_model.eval()

        if self.args.model == "wikitable":
            self.colrel_model = BertForMultiOutputClassification.from_pretrained(
                self.args.shortcut_name,
                num_labels=colrel_num_labels,
                output_attentions=True,
                output_hidden_states=True).to(self.device)
            if self.args.colpair:
                config = BertConfig.from_pretrained(self.args.shortcut_name)
                self.colrel_model.bert.pooler = BertMultiPairPooler(config).to(
                    self.device)

            self.colrel_model.load_state_dict(
                torch.load(colrel_model_path, map_location=self.device))
            self.colrel_model.eval()

    def annotate_columns(self, df: pd.DataFrame):
        adf = AnnotatedDataFrame(df)
        ## Dataset
        input_dataset = DFColTypeTablewiseDataset(df, self.tokenizer)
        input_dataloader = DataLoader(input_dataset,
                                      batch_size=self.args.batch_size,
                                      collate_fn=collate_fn)

        ## Prediction
        batch = next(iter(input_dataloader))

        # 1. Column type
        logits, = self.coltype_model(batch["data"].T)

        outputs = self.coltype_model.bert.encoder(
            self.coltype_model.bert.embeddings(batch["data"].T),
            output_attentions=True,
            output_hidden_states=True)
        hidden_states = outputs[
            1]  # 0: word embeddings, -1: last_hidden_states
        last_hidden_states = outputs.last_hidden_state.squeeze(
            0)  # SeqLen * DimSize

        cls_indexes = torch.nonzero(
            batch["data"].T.squeeze(0) ==
            self.tokenizer.cls_token_id).T.squeeze(0).detach().cpu().numpy()

        emb_list = []
        for cls_id, cls_index in zip(batch["label"].detach().cpu().numpy(),
                                     cls_indexes):
            emb = last_hidden_states[cls_index].squeeze(
                0).detach().cpu().numpy()  # 768
            emb_list.append(emb)

        adf.colemb = emb_list

        if len(logits.shape) == 2:
            logits = logits.unsqueeze(0)

        cls_indexes = torch.nonzero(
            batch["data"].T == self.tokenizer.cls_token_id)
        filtered_logits = torch.zeros(cls_indexes.shape[0],
                                      logits.shape[2]).to(self.device)

        for n in range(cls_indexes.shape[0]):
            i, j = cls_indexes[n]
            logit_n = logits[i, j, :]
            filtered_logits[n] = logit_n

        coltype_pred = filtered_logits.argmax(1)
        if self.args.model == "viznet":
            coltype_pred_labels = [sato_coltypes[x] for x in coltype_pred]
        elif self.args.model == "wikitable":
            coltype_pred_labels = [
                self.coltype_mlb.classes_[x] for x in coltype_pred
            ]

        adf.coltypes = coltype_pred_labels

        # 2. Column relation
        if self.args.model == "wikitable":
            logits, = self.colrel_model(batch["data"].T)
            if len(logits.shape) == 2:
                logits = logits.unsqueeze(0)
            cls_indexes = torch.nonzero(
                batch["data"].T == self.tokenizer.cls_token_id)
            filtered_logits = torch.zeros(cls_indexes.shape[0],
                                          logits.shape[2]).to(self.device)
            for n in range(cls_indexes.shape[0]):
                i, j = cls_indexes[n]
                logit_n = logits[i, j, :]
                filtered_logits[n] = logit_n

            colrel_pred = filtered_logits.argmax(1)
            colrel_pred = colrel_pred[1:]  # Drop the first prediction
            colrel_pred_labels = [
                self.colrel_mlb.classes_[x] for x in colrel_pred
            ]

            adf.colrels = colrel_pred_labels

        return adf


class Dosolo:

    def __init__(self, args=None, basedir="./"):

        self.device = torch.device("cpu")

        if args is None:
            args = argparse.Namespace

        self.args = args
        self.args.shortcut_name = "bert-base-uncased"
        self.args.batch_size = 16

        ## Load models
        coltype_model_path = os.path.join(
            basedir,
            "model/turl_single_bert_bert-base-uncased-bs16-ml-16__turl-1.00_best_micro_f1.pt"
        )

        ## Load mlb
        with open(os.path.join(basedir, "data/turl_coltype_mlb.pickle"),
                  "rb") as fin:
            self.coltype_mlb = pickle.load(fin)

        self.tokenizer = BertTokenizer.from_pretrained(self.args.shortcut_name)
        coltype_config = BertConfig.from_pretrained(self.args.shortcut_name,
                                                    num_labels=255,
                                                    output_attentions=True,
                                                    output_hidden_states=True)
        self.coltype_model = BertForSequenceClassification(coltype_config).to(
            self.device)
        self.coltype_model.load_state_dict(
            torch.load(coltype_model_path, map_location=self.device))
        self.coltype_model.eval()

    def annotate_columns(self, df: pd.DataFrame):
        adf = AnnotatedDataFrame(df)

        ## Dataset
        input_dataset = DFColTypeColwiseDataset(df, self.tokenizer)
        input_dataloader = DataLoader(input_dataset,
                                      batch_size=self.args.batch_size,
                                      collate_fn=collate_fn)

        ## Prediction
        batch = next(iter(input_dataloader))

        # 1. Column type
        outputs = self.coltype_model(batch["data"].T)

        coltype_pred = outputs.logits.argmax(1).detach().cpu().numpy()
        coltype_pred_labels = [
            self.coltype_mlb.classes_[x] for x in coltype_pred
        ]
        adf.coltypes = coltype_pred_labels

        last_hidden_states = outputs.hidden_states[-1][:, 0, :].detach().cpu(
        ).numpy()
        adf.colemb = last_hidden_states

        return adf


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        default="wikitable",
                        type=str,
                        choices=["wikitable", "viznet"],
                        help="Pretrained model")
    parser.add_argument("--input",
                        default=None,
                        type=str,
                        help="Input file (csv)")
    args = parser.parse_args()

    if args.input is None:
        # Sample table
        input_df = pd.read_csv("sample_tables/sample_table1.csv", index_col=0)
    else:
        input_df = pd.read_csv(args.input)

    doduo = Doduo(args)
    annotated_df = doduo.annotate_columns(input_df)

    #dosolo = Dosolo(args)
    #annotated_df = dosolo.annotate_columns(input_df)
