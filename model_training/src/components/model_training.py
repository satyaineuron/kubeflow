import os,sys
import datasets
import pandas as pd
import torch.optim as optim
import numpy as np
from src.constant import *
from src.utils import *
from src.logger import logging
from src.exception import CustomException
from src.utils.custom_model import BERTSentiment
from src.cloud_storage.aws_operations import S3Operation
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact, ModelTrainingArtifact

class ModelTraining:
    def __init__(self, timestamp):
        self.data_ingestion_artifact: DataIngestionArtifact = DataIngestionArtifact(
            timestamp
        )
        self.model_trainer_config: ModelTrainerConfig = ModelTrainerConfig(
            timestamp=timestamp
        )
        self.s3 = S3Operation()

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def initiate_model_trainer(self):
        try:

            train_data = datasets.load_from_disk(self.data_ingestion_artifact.train_file_path)
            test_data = datasets.load_from_disk(self.data_ingestion_artifact.test_file_path)
            valid_data = datasets.load_from_disk(self.data_ingestion_artifact.valid_dir)
            
            OUTPUT_DIM = len(train_data['label'].unique())
            print(OUTPUT_DIM)
            
            model = BERTSentiment(bert, OUTPUT_DIM, FREEZE)

            train_loader = create_dataloader(
                batch_data=train_data, train_value=True)
            valid_loader = create_dataloader(
                batch_data=valid_data, train_value=False)

            best_valid_loss = float('inf')
            optimizer = optim.Adam(model.parameters(), lr=LR)
            criterion = nn.CrossEntropyLoss()

            device = get_default_device()
            model = model.to(device=device)
            criterion = criterion.to(device)

            os.makedirs(self.model_trainer_config.model_training_dir, exist_ok=True)

            train_losses = []
            train_accs = []
            valid_losses = []
            valid_accs = []

            for epoch in range(N_EPOCHS):

                train_loss, train_acc = train(
                    train_loader, model, criterion, optimizer, device)
                valid_loss, valid_acc = evaluate(
                    valid_loader, model, criterion, device)

                train_losses.extend(train_loss)
                train_accs.extend(train_acc)
                valid_losses.extend(valid_loss)
                valid_accs.extend(valid_acc)

                epoch_train_loss = np.mean(train_loss)
                epoch_train_acc = np.mean(train_acc)
                epoch_valid_loss = np.mean(valid_loss)
                epoch_valid_acc = np.mean(valid_acc)

                if epoch_valid_loss < best_valid_loss:
                    best_valid_loss = epoch_valid_loss
                    torch.save(model.state_dict(), self.model_trainer_config.model_trainer_model_file_path)

                print(f'epoch: {epoch+1}')
                print(
                    f'train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}')
                print(
                    f'valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f}')



        except Exception as e:
            raise CustomException(e, sys)