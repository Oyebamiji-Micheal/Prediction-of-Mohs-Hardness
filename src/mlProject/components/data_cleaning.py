import os
from typing import Tuple
from typing_extensions import Annotated 

from mlProject import logger

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler


class FeatureEngineering(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by applying feature engineering.
        
        Args:
            X (pd.DataFrame): The data to transform.
        Returns:
            pd.DataFrame: The transformed data.
        """
        try:
            # Atomic weight and ionization energy ratio
            X["atomicweight_ionenergy_Ratio"] = X["atomicweight_Average"] / (X["ionenergy_Average"] + 0.0000001)

            # Normalized density with respect to the total number of electrons
            X["normalized_density"] = X["density_Total"] / (X["allelectrons_Total"] + 0.0000001)
            
            # Electronegativity and Van der Waals radius ratio
            X["el_neg_chi_R_vdw_Ratio"] = X["el_neg_chi_Average"] / (X["R_vdw_element_Average"] + 0.0000001)
            
            # Number of Electrons Based on Average Atomic Weight
            X["electrons_per_atomicweight"] = X["allelectrons_Average"] / (X["atomicweight_Average"] + 0.0000001)

            # Valence Electron Count
            X["specific_electron_count"] = X["allelectrons_Total"] / (X["atomicweight_Average"] + 0.0000001)

            logger.info("Feature engineering completed successfully")

            return X

        except Exception as e:
            logger.error(f"Feature engineering failed with the following error: {e}")
            raise e
        

class FixOutliers(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by fixing outliers.
        
        Args:
            X (pd.DataFrame): The data to transform.
        Returns:
            pd.DataFrame: The transformed data.
        """
        try:
            # Fixing outliers
            # Note: The following outliers were hard-coded and removed using Tukey's method
            outlier_columns = [
                "allelectrons_Total", "atomicweight_ionenergy_Ratio", "normalized_density", 
                "el_neg_chi_R_vdw_Ratio", "electrons_per_atomicweight", "specific_electron_count"
            ]
            
            for column in outlier_columns:
                IQR = X[column].quantile(0.75) - X[column].quantile(0.25)
                Lower_fence = X[column].quantile(0.25) - (IQR * 3)
                Upper_fence = X[column].quantile(0.75) + (IQR * 3)
                logger.info(f"{column} outliers are values < {Lower_fence} or > {Upper_fence}")
            
            X = X[X[column] <= Upper_fence]

            logger.info("Outliers fixed successfully")

            return X

        except Exception as e:
            logger.error(f"Fixing outliers failed with the following error: {e}")
            raise e
        

class DataPreprocessing(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data by applying data preprocessing.
        
        Args:
            X (pd.DataFrame): The data to transform.
        Returns:
            pd.DataFrame: The transformed data.
        """
        try:
            # Data preprocessing
            X_train = X.drop(columns=["id"], axis=1)

            num_cols = X_train.select_dtypes(include=np.number).columns.to_list()

            scaler = MinMaxScaler()
            matrix = scaler.fit_transform(X_train[num_cols])

            X_train = pd.DataFrame(matrix, columns=num_cols)
            
            logger.info("Data preprocessing completed successfully")

            return X_train

        except Exception as e:
            logger.error(f"Data preprocessing failed with the following error: {e}")
            raise e
