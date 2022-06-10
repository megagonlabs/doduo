import json

import pandas as pd


if __name__ == "__main__":
    train_cnt_dict = json.load(open("sato_78/class_train_count.json"))
    f1_dict = json.load(open("eval/doduo-opendata-sato78_bert-base-uncased_bs-32_maxcollen-64-best-f1micro.json"))[0]
    f1_dict = dict(
        list(
            map(lambda x: (x[0].split("__")[1], x[1]),
                filter(lambda y: "test_f1_class__" in y[0], f1_dict.items()))))

    assert set(f1_dict.keys()) == set(train_cnt_dict.keys())
    names = sorted(train_cnt_dict.keys())
    train_nums = []
    f1_scores = []
    for class_name in names:
        train_nums.append(train_cnt_dict[class_name])
        f1_scores.append(f1_dict[class_name])
    eval_df = pd.DataFrame({"train_count": train_nums,
                            "f1": f1_scores})
    eval_df.index = names
    eval_df = eval_df[eval_df["f1"] >= 0.0]
