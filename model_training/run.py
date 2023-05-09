import os,sys
from src.cloud_storage.aws_operations import S3Operation
from src.components.model_training import ModelTraining
from src.entity.config_entity import TrainingPipelineConfig
from src.logger import logging
from src.exception import CustomException

s3 = S3Operation()

tp = TrainingPipelineConfig()

def start_model_trainer():
    try:
        timestamp = s3.get_pipeline_artifacts(
        bucket_name=tp.artifact_bucket_name, folders=["data_ingestion"]
    )
        model_training =  ModelTraining(timestamp=timestamp)
        model_training.initiate_model_trainer()
    except Exception as e:
        raise CustomException(e, sys)
    
    finally:
        s3.sync_folder_to_s3(
            folder=tp.artifact_dir,
            bucket_name=tp.artifact_bucket_name,
            bucket_folder_name=tp.artifact_dir,
        )
    

if __name__ == "__main__":
    start_model_trainer()