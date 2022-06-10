from functools import reduce
import operator
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import transformers


def collate_fn(samples):
    data = torch.nn.utils.rnn.pad_sequence(
        [sample["data"] for sample in samples])
    label = torch.cat([sample["label"] for sample in samples])
    batch = {"data": data, "label": label}
    if "idx" in samples[0]:
        # For debug purpose
        batch["idx"] = torch.cat([sample["idx"] for sample in samples])
    return batch


class SatoCVColwiseDataset(Dataset):

    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 128,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "./data"):
        if device is None:
            device = torch.device('cpu')

        if multicol_only:
            basename = "msato_cv_{}.csv"
        else:
            basename = "sato_cv_{}.csv"

        assert split in ["train", "valid",
                         "test"], "split must be train or test"

        if split in ["train", "valid"]:
            df_list = []
            for i in range(5):
                if i == cv:
                    continue
                filepath = os.path.join(base_dirpath, basename.format(i))
                df_list.append(pd.read_csv(filepath))
            df = pd.concat(df_list, axis=0)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)

        # [CLS] [SEP] will be automatically added, so max_length should be +2
        # TODO: This will be different, depending on how many columns to have

        # For learning curve
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)

        row_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                break
            if split == "valid" and i < valid_index:
                continue

            for _, row in group_df.iterrows():
                row_list.append(row)

        self.df = pd.DataFrame(row_list)

        # Convert into torch.Tensor
        self.df["data_tensor"] = self.df["data"].apply(
            lambda x: torch.LongTensor(
                tokenizer.encode(x,
                                 add_special_tokens=True,
                                 max_length=max_length + 2)).to(device))
        self.df["label_tensor"] = self.df["class_id"].apply(
            lambda x: torch.LongTensor([x]).to(device)
        )  # Can we reduce the size?

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"]
        }


class SatoCVTablewiseDataset(Dataset):

    def __init__(
            self,
            cv: int,
            split: str,  # train or test
            tokenizer: transformers.PreTrainedTokenizer,
            max_length: int = 128,
            multicol_only: bool = False,
            train_ratio: float = 1.0,
            device: torch.device = None,
            base_dirpath: str = "./data"):
        if device is None:
            device = torch.device('cpu')

        assert split in ["train", "valid",
                         "test"], "split must be train or test"

        if multicol_only:
            basename = "msato_cv_{}.csv"
        else:
            basename = "sato_cv_{}.csv"

        if split in ["train", "valid"]:
            df_list = []
            for i in range(5):
                if i == cv:
                    continue
                filepath = os.path.join(base_dirpath, basename.format(i))
                df_list.append(pd.read_csv(filepath))
            df = pd.concat(df_list, axis=0)
        else:
            # test
            filepath = os.path.join(base_dirpath, basename.format(cv))
            df = pd.read_csv(filepath)

        # [CLS] [SEP] will be automatically added, so max_length should be +2
        # TODO: This will be different, depending on how many columns to have

        # For learning curve
        num_tables = len(df.groupby("table_id"))
        valid_index = int(num_tables * 0.8)
        num_train = int(train_ratio * num_tables * 0.8)

        data_list = []
        for i, (index, group_df) in enumerate(df.groupby("table_id")):
            if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                break
            if split == "valid" and i < valid_index:
                continue

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
                group_df["class_id"].values).to(device)
            data_list.append(
                [index,
                 len(group_df), token_ids, class_ids, cls_indexes])

        self.table_df = pd.DataFrame(data_list,
                                     columns=[
                                         "table_id", "num_col", "data_tensor",
                                         "label_tensor", "cls_indexes"
                                     ])
        """
        # NOTE: msato contains a small portion of single-col tables. keep it to be consistent.  
        if multicol_only:
            # Check
            num_all_tables = len(self.table_df)
            self.table_df = self.table_df[self.table_df["num_col"] > 1]
            assert len(self.table_df) == num_all_tables
        """

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}


class TURLColTypeColwiseDataset(Dataset):
    """TURL column type prediction column-wise (single-column)"""

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]
        self.mlb = df_dict["mlb"]  # MultilabelBinarizer model

        # For learning curve
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)

        row_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break

            # [WARNING] This potentially affects the evaluation results as well
            if split == "train" and len(group_df) > max_colnum:
                continue

            for _, row in group_df.iterrows():
                row_list.append(row)

        self.df = pd.DataFrame(row_list)
        self.df["data_tensor"] = self.df["data"].apply(
            lambda x: torch.LongTensor(
                tokenizer.encode(
                    x, add_special_tokens=True, max_length=max_length + 2)).to(
                        device)).tolist()

        self.df["label_tensor"] = self.df["label_ids"].apply(
            lambda x: torch.LongTensor([x]).to(device))

        if multicol_only:
            # Do nothing
            pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"]
        }


class TURLColTypeTablewiseDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]
        self.mlb = df_dict["mlb"]  # MultilabelBinarizer model

        # For learning curve
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)

        data_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break

            # [WARNING] This potentially affects the evaluation results as well
            if split == "train" and len(group_df) > max_colnum:
                continue

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

        if multicol_only:
            self.table_df = self.table_df[self.table_df["num_col"] > 1]

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }


class TURLRelExtColwiseDataset(Dataset):
    """TURL column relation prediction column-wise (single-column)"""

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]
        self.mlb = df_dict["mlb"]  # MultilabelBinarizer model

        # For learning curve
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)

        row_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break

            # [WARNING] This potentially affects the evaluation results as well
            if split == "train" and len(group_df) > max_colnum:
                continue

            group_df = group_df.sort_values("column_id")

            for j, (_, row) in enumerate(group_df.iterrows()):
                if j == 0:
                    continue

                row["data_tensor"] = torch.LongTensor(
                    tokenizer.encode(group_df.iloc[0]["data"],
                                     add_special_tokens=True,
                                     max_length=max_length + 2) +
                    tokenizer.encode(row["data"],
                                     add_special_tokens=True,
                                     max_length=max_length + 2)).to(device)

                row_list.append(row)

        self.df = pd.DataFrame(row_list)
        self.df["label_tensor"] = self.df["label_ids"].apply(
            lambda x: torch.LongTensor([x]).to(device))

        if multicol_only:
            # Do nothing
            pass

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return {
            "data": self.df.iloc[idx]["data_tensor"],
            "label": self.df.iloc[idx]["label_tensor"]
        }
        #"idx": torch.LongTensor([idx])}
        #"cls_indexes": self.table_df.iloc[idx]["cls_indexes"]}


class TURLRelExtTablewiseDataset(Dataset):

    def __init__(self,
                 filepath: str,
                 split: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_length: int = 32,
                 max_colnum: int = 10,
                 multicol_only: bool = False,
                 train_ratio: float = 1.0,
                 device: torch.device = None):
        if device is None:
            device = torch.device('cpu')

        with open(filepath, "rb") as fin:
            df_dict = pickle.load(fin)

        assert split in df_dict
        self.df = df_dict[split]
        self.mlb = df_dict["mlb"]  # MultilabelBinarizer model

        # For learning curve
        num_tables = len(self.df.groupby("table_id"))
        num_train = int(train_ratio * num_tables)

        data_list = []
        for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
            if i >= num_train:
                break

            # It's probably already sorted but just in case.
            group_df = group_df.sort_values("column_id")

            # [WARNING] This potentially affects the evaluation results as well
            if split == "train" and len(group_df) > max_colnum:
                continue

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

        if multicol_only:
            self.table_df = self.table_df[self.table_df["num_col"] > 1]

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"]
        }
