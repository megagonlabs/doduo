from collections import Counter
import json

import pandas as pd

if __name__ == "__main__":
    cnt = Counter()
    df = pd.read_csv("sato_78/table_dbpedia_sato_78_all.csv")
    for index, row in df.iterrows():
        data = json.loads(row["content"])
        for k, v in data.items():
            if "annotation_label" in v:
                cnt[v["annotation_label"]] += 1

    class_names = sorted(list(dict(cnt).keys()))
    class_index_dict = dict([[name, i] for i, name in enumerate(class_names)])

    with open("sato_78/class_index.json", "w") as fout:
        json.dump(class_index_dict, fout)

    #
    train_cnt = Counter()
    train_df = pd.read_csv("sato_78/train_table_dbpedia_sato_78.csv")
    for index, row in df.iterrows():
        data = json.loads(row["content"])
        for k, v in data.items():
            if "annotation_label" in v:
                train_cnt[v["annotation_label"]] += 1
    
    with open("sato_78/class_train_count.json", "w") as fout:
        json.dump(dict(train_cnt), fout)
