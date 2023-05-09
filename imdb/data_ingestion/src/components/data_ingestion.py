import os,sys
import pandas as pd
import datasets
from src.exception import CustomException
from src.constant import *
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

class DataIngestion:

    def __init__(self):
        self.data_ingestion_config: DataIngestionConfig = DataIngestionConfig()

    def tokenize_and_numericalize(self, sentence, tokenizer):
            """
            Tokenizes and numericalizes a given sentence using a specified tokenizer.
            """
            try:
                ids = tokenizer(sentence['text'], truncation=True)['input_ids']
                return {'ids': ids}
            except Exception as e:
                raise CustomException(e, sys)
            

    def initiate_data_ingestion(self):
            try:
                train_data, test_data = datasets.load_dataset(
                'imdb', split=['train', 'test'])

                train_data = train_data.map(self.tokenize_and_numericalize, fn_kwargs={
                                        'tokenizer': tokenizer})
                test_data = test_data.map(self.tokenize_and_numericalize, fn_kwargs={
                                      'tokenizer': tokenizer})
                
                # splitting train data into train(75%) and valid(25%)
                train_valid_data = train_data.train_test_split(test_size=TEST_SIZE)
                train_data = train_valid_data['train']
                valid_data = train_valid_data['test']

                train_data = train_data.with_format(
                    type='torch', columns=['ids', 'label'])
                test_data = test_data.with_format(
                    type='torch', columns=['ids', 'label'])
                valid_data = valid_data.with_format(
                    type='torch', columns=['ids', 'label'])
                
                os.makedirs(self.data_ingestion_config.feature_store_dir, exist_ok=True)

                train_data.save_to_disk(self.data_ingestion_config.training_file_path)
                test_data.save_to_disk(self.data_ingestion_config.testing_file_path)
                valid_data.save_to_disk(self.data_ingestion_config.valid_file_path)

                data_ingestion_artifact: DataIngestionArtifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_dir
            )
                return data_ingestion_artifact

            except Exception as e:
                    raise CustomException(e, sys)
