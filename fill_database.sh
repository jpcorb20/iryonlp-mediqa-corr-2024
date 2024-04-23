MAIN_DIR=/home/Users/Jean-Philippe.Corbeil

# For local server instance: ANONYMIZED_TELEMETRY=False chroma run --path YOUR_PATH
# Empty env var for separate running instance.
CHROMA_PATH=/chromadb

# Encoders:
# - MedCPT-Query-Encoder
# - MedCPT-Article-Encoder
# - pubmedbert-base-embeddings-matryoshka
EMBEDER=./embeddings/MedCPT-Query-Encoder

# Use -fs arg for $FEW_SHOTS.
# FEW_SHOTS=$MAIN_DIR/mediqa-corr-llm//datasets/Feb_1_2024_MS_Train_Val_Datasets/MEDIQA-CORR-2024-MS-TrainingData-UTF.csv
# --OR--
# Use -mw arg for any ClinicalCorp dataset.
MEDWIKI=$MAIN_DIR/datasets/medwiki/med_wiki

python -m fill_db -p "$CHROMA_PATH" -mw $MEDWIKI -em $EMBEDER
