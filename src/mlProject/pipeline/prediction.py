from pathlib import Path

import numpy as np
import pandas as pd
import joblib 


class PredictionPipeline:
    def __init__(self):
        self.transformation = joblib.load(Path(
            'artifacts/data_transformation/transformation_pipeline.joblib'
        )) 
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    
    def predict(self, data):
        preprocessed_data = self.transformation.transform(data)

        print(preprocessed_data)
        
        prediction = self.model.predict(preprocessed_data)

        return prediction