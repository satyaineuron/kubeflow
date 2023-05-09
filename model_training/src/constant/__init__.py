import transformers
import torch.nn as nn
from datetime import datetime
from transformers import BertConfig


TIMESTAMP: datetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACT_DIR: str = "artifacts"

APP_ARTIFACTS_BUCKET : str = "imdb-artifacts"

LOG_DIR: str = "logs"

# Data Ingestion
DATA_INGESTION_DIR_NAME: str = "data_ingestion"

DATA_INGESTION_FEATURE_STORE_DIR : str = 'feature_store'

TEST_SIZE = 0.25

TRAIN_FILE_NAME: str = "sam.csv"

TEST_FILE_NAME: str = "test.csv"

VALID_FILE_NAME: str = "valid.csv"

# Model Training
MODEL_NAME = 'bert-base-uncased'
BATCH_SIZE = 8
HIDDEN_DIM = 256
FREEZE = False
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25
N_EPOCHS = 5
LR = 1e-5
criterion = nn.CrossEntropyLoss()

OUTPUT_DIM =2

MODEL_TRAINING_DIR: str = "model_training"


config = BertConfig.from_pretrained(MODEL_NAME)
bert = transformers.AutoModel.from_pretrained(MODEL_NAME, config=config)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)