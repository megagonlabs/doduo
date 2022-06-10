import argparse

import pandas as pd

from doduo.doduo import Doduo

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="wikitable",
        type=str,
        choices=["wikitable", "viznet"],
        help="Pretrained model"
    )
    parser.add_argument(
        "--input",
        default=None,
        type=str,
        help="Input file (csv)"
    )
    args = parser.parse_args()

    if args.input is None:
        # Sample table
        input_df = pd.read_csv(
            "sample_tables/sample_table1.csv",
            index_col=0)
    else:
        input_df = pd.read_csv(args.input)

    doduo = Doduo(args)
    annotated_df = doduo.annotate_columns(input_df)