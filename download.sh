#!/bin/bash

# Datasets in this repo include variations of the "Sato" and "TURL" datasets.

# Sato refers to the dataset used in ["Sato: Contextual Semantic Type Detection in Tables." Proceedings of the VLDB Endowment Vol. 13, No.11](https://github.com/megagonlabs/sato).
# The dataset was generated from the [VizNet](https://github.com/mitmedialab/viznet) corpus.

# TURL refers to the dataset used in ["TURL: table understanding through representation learning." Proceedings of the VLDB Endowment 14.3 (2020): 307-319](https://github.com/sunlab-osu/TURL). The dataset was generated from the [WikiTable](http://websail-fe.cs.northwestern.edu/TabEL/) corpus.

wget https://doduo-data.s3-us-west-2.amazonaws.com/data.tar.gz
tar -zvxf data.tar.gz

# Pretrained models
wget https://doduo-data.s3-us-west-2.amazonaws.com/model.tar.gz
tar -zvxf model.tar.gz

