set -a # automatically export all variables
source .env
set +a

MAIN_DIR=/home/Users/Jean-Philippe.Corbeil/mediqa-corr-2024/datasets
FILE=$MAIN_DIR/Feb_1_2024_MS_Train_Val_Datasets/MEDIQA-CORR-2024-MS-ValidationSet-1-Full-UTF.csv
# No path for client instance on localhost:8000
CHROMA_PATH=""

nohup python -m reflexion -f $FILE -o reflexion_results -db $CHROMA_PATH -em ./embeddings/pubmedbert-base-embeddings-matryoshka > reflexion.log 2>&1 &
