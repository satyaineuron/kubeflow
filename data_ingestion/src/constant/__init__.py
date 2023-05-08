import transformers
from datetime import datetime
from transformers import BertConfig


TIMESTAMP: datetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACT_DIR: str = "artifacts"

APP_ARTIFACTS_BUCKET : str = "imdb-artifacts"

LOG_DIR: str = "logs"

DATA_INGESTION_DIR_NAME: str = "data_ingestion"

MODEL_NAME = 'bert-base-uncased'

TEST_SIZE = 0.25

DATA_INGESTION_FEATURE_STORE_DIR : str = 'feature_store'

TRAIN_FILE_NAME: str = "train.csv"

TEST_FILE_NAME: str = "test.csv"

VALID_FILE_NAME: str = "valid.csv"

config = BertConfig.from_pretrained(MODEL_NAME)
bert = transformers.AutoModel.from_pretrained(MODEL_NAME, config=config)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)

