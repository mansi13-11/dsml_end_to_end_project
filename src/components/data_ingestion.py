import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
@dataclass

class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        # self.data_path = self.ingestion_config.raw_data_path
        # self.test_size = 0.2
        # self.random_state = 42


    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            # Read the dataset
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Dataset read successfully")

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data
            # raw_data_path = os.path.join(os.path.dirname(self.data_path), "raw_data.csv")
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved")

            # Split the data into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data split into training and testing sets")

            # # Save the training and testing sets
            # train_data_path = os.path.join(os.path.dirname(self.data_path), "train_data.csv")
            # test_data_path = os.path.join(os.path.dirname(self.data_path), "test_data.csv")

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Training data saved")
            logging.info("Testing data saved")

            return(self.ingestion_config.train_data_path, self.ingestion_config.test_data_path)

        except Exception as e:
            logging.error("Error occurred during data ingestion")
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()