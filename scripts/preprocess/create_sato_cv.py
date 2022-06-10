import os
import re

import pandas as pd
from tqdm import tqdm

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

sato_coltype_id_dict = dict([(name, i) for i, name in enumerate(sato_coltypes)])


def canonical_header(h, max_header_len=30):
    # convert any header to its canonincal form
    # e.g. fileSize
    h = str(h)
    if len(h)> max_header_len:
        return '-'
    h = re.sub(r'\([^)]*\)', '', h) # trim content in parentheses
    h = re.sub(r"([A-Z][a-z])", r" \1", h) #insert a space before any Cpital starts
    words = list(filter(lambda x: len(x)>0, map(lambda x: x.lower(), re.split('\W', h))))
    if len(words)<=0:
        return '-'
    new_phrase = ''.join([words[0]] + [x.capitalize() for x in words[1:]])
    return new_phrase


def load_filtered(fileName):
    df = pd.read_csv(fileName)
    df = df.rename(columns=lambda x: canonical_header(x)) # canonicalized header

    # Note: This raises error if the DataFrame has duplicate column names
    # df = df.filter(items=sato_coltypes, axis=1) # filte columns
    df = df.loc[:, df.columns.isin(sato_coltypes)]

    return df
# example use, load partition #0 for webtables1
# the returned dataframe will have canonical columns types within the 78 types


if __name__ == "__main__":
    """
    for cv in tqdm(range(5)):
        data_list = []
        for corpus in ["webtables1", "webtables2"]:
            dirpath = "../extracted_tables_cv/{}/K{}/".format(corpus, cv)
            for filename in os.listdir(dirpath):
                filepath = os.path.join(dirpath, filename)
                df = load_filtered(filepath)
                for i in range(len(df.columns)):
                    if len(df.iloc[:, i].dropna()) > 0:
                        # At least one non-NaN data should be there
                        data_list.append(["{}/K{}/{}".format(corpus, cv, filename),
                                          i,
                                          df.columns[i],
                                          sato_coltype_id_dict[df.columns[i]],
                                          " ".join([str(x) for x in df.iloc[:, i].dropna().tolist()])])
        df = pd.DataFrame(data_list, columns=["table_id", "col_idx", "class", "class_id", "data"])
        df.to_csv("data/sato_cv_{}.csv".format(cv), index=False)
    """

    for cv in tqdm(range(5)):
        data_list = []
        for corpus in ["webtables1", "webtables2"]:
            dirpath = "../extracted_tables_cv/{}/K{}_multi-col/".format(corpus, cv)
            for filename in os.listdir(dirpath):
                filepath = os.path.join(dirpath, filename)
                df = load_filtered(filepath)
                for i in range(len(df.columns)):
                    if len(df.iloc[:, i].dropna()) > 0:
                        # At least one non-NaN data should be there
                        data_list.append(["{}/K{}_multi-col/{}".format(corpus, cv, filename),
                                          i,
                                          df.columns[i],
                                          sato_coltype_id_dict[df.columns[i]],
                                          " ".join([str(x) for x in df.iloc[:, i].dropna().tolist()])])
        df = pd.DataFrame(data_list, columns=["table_id", "col_idx", "class", "class_id", "data"])
        df.to_csv("data/msato_cv_{}.csv".format(cv), index=False)

