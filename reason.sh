set -a # automatically export all variables
source .env
set +a

MAIN_DIR=/home/Users/Jean-Philippe.Corbeil/mediqa-corr-2024/datasets/Feb_1_2024_MS_Train_Val_Datasets
FILE=$MAIN_DIR/MEDIQA-CORR-2024-MS-TrainingData.csv
python -m reasoning -f $FILE -o "train_reasoning"
