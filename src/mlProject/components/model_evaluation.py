from pathlib import Path

from mlProject import logger
from mlProject.utils.common import save_json
from mlProject.entity.config_entity import ModelEvaluationConfig

import numpy as np
import pandas as pd
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_absolute_error, r2_score
import joblib


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    
    def eval_metrics(self, actual, pred):
        median_ae = median_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mean_ae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return median_ae, rmse, mean_ae, r2
    

    def save_results(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        logger.info("Model loaded successfully")

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        predicted_hardness = model.predict(test_x)
        (median_ae, rmse, mean_ae, r2) = self.eval_metrics(test_y, predicted_hardness)

        logger.info("Evaluation metrics calculated successfully")

        # Saving metrics 
        scores = {"median_ae": median_ae, "rmse": rmse, "mean_ae": mean_ae, "r2": r2}
        save_json(path=Path(self.config.metric_file_name), data=scores)

        logger.info("Evaluation metrics saved successfully")
