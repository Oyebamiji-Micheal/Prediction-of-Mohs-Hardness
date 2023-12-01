import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from mlProject import logger
from mlProject.entity.config_entity import DataTransformationConfig
from mlProject.components.data_cleaning import FeatureEngineering, FixOutliers, DataPreprocessing



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def data_transformation_pipeline(self):
        """Create a pipeline for data transformation

        Args:
            config (DataTransformationConfig): Data transformation configuration
        Returns:
            pipeline: Pipeline for data transformation
        """
        
        try:
            pipeline = Pipeline([
                ("Feature Engineering", FeatureEngineering()),
                ("Fix Outliers", FixOutliers()),
                ("Data Preprocessing", DataPreprocessing()),
            ])

            return pipeline
        except Exception as e:
            logger.error(f"Data transformation failed with the following error: {e}")
            raise e

    def transform_split_data(self):
        """Transform and split data into train and test sets
        
        Args:
            config (DataTransformationConfig): Data transformation configuration
        Returns:
            None
        """

        data = pd.read_csv(self.config.data_path)
        pipeline = self.data_transformation_pipeline()

        processed_data = pipeline.fit_transform(data)
        
        # Split data into train and test sets
        train, test = train_test_split(processed_data, test_size=0.2, random_state=42)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Pipeline transformation and data splitting completed successfully")

        logger.info(train.shape)
        logger.info(test.shape)        
