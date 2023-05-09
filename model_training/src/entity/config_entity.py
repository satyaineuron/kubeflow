import os

from src.constant import *

class TrainingPipelineConfig:
    def __init__(self):
        self.artifact_dir: str = ARTIFACT_DIR

        self.artifact_bucket_name: str = APP_ARTIFACTS_BUCKET

class ModelTrainerConfig:
    def __init__(self, timestamp):
        self.model_training_dir: str = os.path.join(
            ARTIFACT_DIR,
            timestamp,
            MODEL_TRAINING_DIR,
        )

        self.model_trainer_model_file_path: str = os.path.join(
            self.model_training_dir,MODEL_TRAINER_MODEL_FILE_PATH
        )