set -a # automatically export all variables
source .env
set +a

MAIN_DIR=/home/Users/Jean-Philippe.Corbeil/mediqa-corr-2024/datasets/
FEW_SHOTS=$MAIN_DIR/chromadb
FILE=$MAIN_DIR/Feb_1_2024_MS_Train_Val_Datasets/MEDIQA-CORR-2024-MS-ValidationSet-1-Full-UTF.csv
python -m main -f $FILE -fs $FEW_SHOTS -n 3 -o "results/output_s50_fs3_noreasoning" -s 50
