# /bin/bash

WORKDIR=/home/Users/Jean-Philippe.Corbeil/mediqa-corr-2024
SUBMISSION_FILE=$WORKDIR/results.txt
REF_CSV=$WORKDIR/datasets/Feb_1_2024_MS_Train_Val_Datasets/MEDIQA-CORR-2024-MS-ValidationSet-1-Full.csv
MODEL_DIR=$WORKDIR/embeddings

python -m evaluation.cli -f $SUBMISSION_FILE -c $REF_CSV -m $MODEL_DIR -fr
