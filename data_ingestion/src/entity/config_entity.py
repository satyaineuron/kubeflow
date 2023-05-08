import os

from src.constant import *

class TrainingPipelineConfig:
    def __init__(self):
        self.artifact_dir: str = ARTIFACT_DIR

        self.artifact_bucket_name: str = APP_ARTIFACTS_BUCKET

class DataIngestionConfig:
    def __init__(self):
        self.data_ingestion_dir: str = os.path.join(
            ARTIFACT_DIR + "/" + TIMESTAMP
        )
    
        self.feature_store_dir: str = os.path.join(
                self.data_ingestion_dir,
                "data_ingestion",
                DATA_INGESTION_FEATURE_STORE_DIR,
            )
        
        self.training_file_path: str = os.path.join(
            self.feature_store_dir,
            TRAIN_FILE_NAME,
        )

        self.testing_file_path: str = os.path.join(
            self.feature_store_dir,
            TEST_FILE_NAME,
        )

        self.valid_file_path: str = os.path.join(
            self.feature_store_dir,
            VALID_FILE_NAME,
        )