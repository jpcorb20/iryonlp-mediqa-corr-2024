set -a # automatically export all variables
source .env
set +a

MAIN_DIR=/home/Users/Jean-Philippe.Corbeil/mediqa-corr-2024/datasets
FILE=$MAIN_DIR/Feb_1_2024_MS_Train_Val_Datasets/MEDIQA-CORR-2024-MS-ValidationSet-1-Full-UTF.csv
CHROMA_PATH=$MAIN_DIR/chromadb
python -m react -f $FILE -o react_res_all -db $CHROMA_PATH  -em ./embeddings/pubmedbert-base-embeddings-matryoshka
