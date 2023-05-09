import os
from dataclasses import dataclass
from src.constant import *

class DataIngestionArtifact:
    def __init__(self, timestamp):

        self.train_file_path: str = os.path.join(
            ARTIFACT_DIR,
            timestamp,
            DATA_INGESTION_DIR_NAME,
            DATA_INGESTION_FEATURE_STORE_DIR,
            TRAIN_DIR
        )
        self.test_file_path: str = os.path.join(
            ARTIFACT_DIR,
            timestamp,
            DATA_INGESTION_DIR_NAME,
            DATA_INGESTION_FEATURE_STORE_DIR,
            TEST_DIR
        )
        self.valid_dir: str = os.path.join(
            ARTIFACT_DIR,
            timestamp,
            DATA_INGESTION_DIR_NAME,
            DATA_INGESTION_FEATURE_STORE_DIR,
            VALID_DIR
        )

class ModelTrainingArtifact:
    pass