
  

# doduo

  

This repository contains the codebase of our paper [Annotating Columns with Pre-trained Language Models](https://arxiv.org/abs/2104.01785), available at arXiv and appearing at SIGMOD 2022.



## Installation

  

```console
$ git clone [link to repo]
$ cd doduo
$ pip install -r requirements.txt 
```

  

With `conda`, create a virtual environment and install the required packages as below:

  

```console
$ conda create --name doduo python=3.7.10
$ conda activate doduo
$ pip install -r requirements.txt
```

  

  

## Data Preparation

  

Run `download.sh` to download processed datasets for the VizNet corpus.
It will also create `data` directory.

  
```console
$ bash download.sh

```

  

You will see the following files in `data`

  

```console
$ ls data
msato_cv_0.csv
msato_cv_1.csv
msato_cv_2.csv
msato_cv_3.csv
msato_cv_4.csv
sato_cv_0.csv
sato_cv_1.csv
sato_cv_2.csv
sato_cv_3.csv
sato_cv_4.csv
table_col_type_serialized.pkl
table_rel_extraction_serialized.pkl
```

  

  

For the WikiTable corpus, download the following files from [here](https://github.com/sunlab-osu/TURL#data) and save them under `data/turl_dataset`.

  

```console
$ tree data/turl_dataset
data/turl_dataset
├── dev.table_col_type.json
├── dev.table_rel_extraction.json
├── test.table_col_type.json
├── test.table_rel_extraction.json
├── train.table_col_type.json
└── train.table_rel_extraction.json
```

  

## Training

  

### Usage

  

```console
optional arguments:
  -h, --help            show this help message and exit
  --shortcut_name SHORTCUT_NAME
                        Huggingface model shortcut name
  --max_length MAX_LENGTH
                        The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
  --batch_size BATCH_SIZE
                        Batch size
  --epoch EPOCH         Number of epochs for training
  --random_seed RANDOM_SEED
                        Random seed
  --num_classes NUM_CLASSES
                        Number of classes
  --multi_gpu           Use multiple GPU
  --fp16                Use FP16
  --warmup WARMUP       Warmup ratio
  --lr LR               Learning rate
  --tasks {sato0,sato1,sato2,sato3,sato4,msato0,msato1,msato2,msato3,msato4,turl,turl-re,turl-sch,turl-re-sch} [{sato0,sato1,sato2,sato3,sato4,msato0,msato1,msato2,msato3,msato4,turl,turl-re,turl-sch,turl-re-sch} ...]
                        Task name {sato, turl, turl-re, turl-sch, turl-re-sch}
  --colpair             Use column pair embedding
  --train_ratios TRAIN_RATIOS [TRAIN_RATIOS ...]
                        e.g., --train_ratios turl=0.8 turl-re=0.1
  --from_scratch        Training from scratch
  --single_col          Training with single column model
```
  

## Training

  

Training the full doduo model:

```console

$ python doduo/train_multi.py --tasks turl turl_re-colpair --max_length 32 --batch_size 16

```

  

To specify GPU, use `CUDA_VISIBLE_DEVICES` environment variable. For example,

  

```console
$ CUDA_VISIBLE_DEVICES=0 python doduo/train_multi.py --tasks turl --max_length 32 --batch_size 16
```

  

After training, you will see the following files in the `./model` directory.

  

```console
$ ls model
```

  

  

## Prediction

  

### Usage

  

```console
$ python doduo/predict_multi.py <model_path>
```

  

### Example

  

```console
$ python doduo/predict_multi.py model/turl_mosato_bert_bert-base-uncased-bs16-ml-32__turl-1.00
```

Note while specifying model names, the last part of the model name "\_best\_macro\_f1.pt" needs to be omitted.

  

  

### Output

  

The inference code will produce a JSON file in `./eval`.

  

## Annotating pandas dataframe

  

The Doduo Python module uses Pandas DataFrame as the base data structure. You can annotate any tables (i.e., only columns at this moment)

  

``` Python
doduo = Doduo()
df = pd.read_csv("your_table.csv")
annotated_df = doduo.annotate_columns(df)
```

  

Doduo.annotate_columns() will annotate the following three attributes to the input table and return an AnnotatedDataFrame object.

* coltypes (List[str])): Predicted column types
* colrels (List[str]): Predicted column relations
* colemb (List[np.ndarray]): Contextualized column embeddings



Let's take a look at examples.

  

``` Python
import argparse
import pandas as pd
from doduo import Doduo


# Load Doduo model
args = argparse.Namespace
args.model = "wikitable" # or args.model = "viznet"
doduo = Doduo(args)

  
# Load sample tables
df1 = pd.read_csv("sample_tables/sample_table1.csv", index_col=0)
df2 = pd.read_csv("sample_tables/sample_table2.csv", index_col=0)

  

# Sample 1: Column annotation
annot_df1 = doduo.annotate_columns(df1)
print(annot_df1.coltypes)

  

# Sample 2: Column annotation
annot_df2 = doduo.annotate_columns(df2)
print(annot_df2.coltypes)
```

  

## Citation

  

```
@inbook{10.1145/3514221.3517906,
author = {Suhara, Yoshihiko and Li, Jinfeng and Li, Yuliang and Zhang, Dan and Demiralp, \c{C}a\u{g}atay and Chen, Chen and Tan, Wang-Chiew},
title = {Annotating Columns with Pre-trained Language Models},
year = {2022},
isbn = {9781450392495},
publisher = {Association for Computing Machinery},
url = {https://doi.org/10.1145/3514221.3517906},
booktitle = {Proceedings of the 2022 International Conference on Management of Data},
}
```

  

  

## Acknowledgment

Datasets in this repo include variations of the "Sato" and "TURL" datasets.

Sato refers to the dataset used in ["Sato: Contextual Semantic Type Detection in Tables." Proceedings of the VLDB Endowment Vol. 13, No.11](https://github.com/megagonlabs/sato). The dataset was generated from the [VizNet](https://github.com/mitmedialab/viznet) corpus.

URL refers to the dataset used in ["TURL: table understanding through representation learning." Proceedings of the VLDB Endowment 14.3 (2020): 307-319](https://github.com/sunlab-osu/TURL). The dataset was generated from the [WikiTable](http://websail-fe.cs.northwestern.edu/TabEL/) corpus.

  

  

# Disclosure

Embedded in, or bundled with, this product are open source software (OSS) components, datasets and other third party components identified below. The license terms respectively governing the datasets and third-party components continue to govern those portions, and you agree to those license terms, which, when applicable, specifically limit any distribution. You may receive a copy of, distribute and/or modify any open source code for the OSS component under the terms of their respective licenses. In the event of conflicts between Megagon Labs, Inc. Recruit Co., Ltd., license conditions and the Open Source Software license conditions, the Open Source Software conditions shall prevail with respect to the Open Source Software portions of the software. You agree not to, and are not permitted to, distribute actual datasets used with the OSS components listed below. You agree and are limited to distribute only links to datasets from known sources by listing them in the datasets overview table below. You are permitted to distribute derived datasets of data sets from known sources by including links to original dataset source in the datasets overview table below. You agree that any right to modify datasets originating from parties other than Megagon Labs, Inc. are governed by the respective third party’s license conditions. All OSS components and datasets are distributed WITHOUT ANY WARRANTY, without even implied warranty such as for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE, and without any liability to or claim against any Megagon Labs, Inc. entity other than as explicitly documented in this README document. You agree to cease using any part of the provided materials if you do not agree with the terms or the lack of any warranty herein. While Megagon Labs, Inc., makes commercially reasonable efforts to ensure that citations in this document are complete and accurate, errors may occur. If you see any error or omission, please help us improve this document by sending information to contact_oss@megagon.ai.

All dataset and code used within the product are listed below (including their copyright holders and the license conditions). For Datasets having different portions released under different licenses, please refer to the included source link specified for each of the respective datasets for identifications of dataset files released under the identified licenses.

 

  

| ID  | OSS Component Name | Modified | Copyright Holder | Upstream Link | License  |
|-----|----------------------------------|----------|------------------|-----------------------------------------------------------------------------------------------------------|--------------------|
| 1 | BERT model | Yes  | Hugging Face | [link](https://huggingface.co/transformers/v3.0.2/_modules/transformers/modeling_bert.html) | Apache License 2.0 |

  

  

| ID  | Dataset | Modified | Copyright Holder | Source Link  | License |
|-----|---------------------|----------|------------------------|----------------------------------------------------------|---------|
| 1 | Sato dataset | No  |  | [source](https://github.com/megagonlabs/sato) generated from [VizNet](https://github.com/mitmedialab/viznet) | |
| 2 | TURL dataset | No  |  | [source](https://github.com/sunlab-osu/TURL)  generated from [WikiTable](http://websail-fe.cs.northwestern.edu/TabEL/)| |
