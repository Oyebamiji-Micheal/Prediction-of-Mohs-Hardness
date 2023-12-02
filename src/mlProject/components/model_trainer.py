import os

from mlProject import logger
from mlProject.entity.config_entity import ModelTrainerConfig

import pandas as pd
from lightgbm import LGBMRegressor
import joblib


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        X_train = train_data.drop([self.config.target_column], axis=1)
        X_test = test_data.drop([self.config.target_column], axis=1)
        y_train = train_data[[self.config.target_column]]
        y_test = test_data[[self.config.target_column]]

        lgbm_reg = LGBMRegressor(max_depth=self.config.max_depth,
                                 num_leaves=self.config.num_leaves,
                                 min_child_samples=self.config.min_child_samples,
                                 learning_rate=self.config.learning_rate,
                                 n_estimators=self.config.n_estimators,
                                 min_child_weight=self.config.min_child_weight,
                                 subsample=self.config.subsample,
                                 colsample_bytree=self.config.colsample_bytree,
                                 reg_alpha=self.config.reg_alpha,
                                 reg_lambda=self.config.reg_lambda,
                                 random_state=self.config.random_state,
                                 extra_trees=self.config.extra_trees
        )

        logger.info("Training model...")

        lgbm_reg.fit(X_train, y_train)

        logger.info("Model trained successfully!")

        joblib.dump(lgbm_reg, os.path.join(self.config.root_dir, self.config.model_name))

        logger.info("Model saved successfully!")
