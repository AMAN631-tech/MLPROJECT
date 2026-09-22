import os
import sys
from dataclasses import dataclass 

import numpy as np  
import pandas as pd 
from sklearn.impute import SimpleImputer    
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder

from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object

@dataclass
class DataTransformationConfiq:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")   


class DataTransformation:
    def __init__(self):
        self.data_transformation_confiq=DataTransformationConfiq()

    def get_datatransformer_object(self):
        """
        this function is resposible for data transformation.
        """
        try:
            numerical_columns = ["reading score","writing score"]
            categorical_columns = [
                "gender",
                "race/ethnicity",
                "parental level of education",
                "lunch",
                "test preparation course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="median")),
                    ("Scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="most_frequent")),
                    ("Encoder",OneHotEncoder()),
                    ("Scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical Coluumns: {numerical_columns}")
            logging.info(f"Categorical Coluumns: {categorical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys) 

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("train and test data readed completly")
            logging.info("Obtaining preprocessing Object")

            preprocessing_obj=self.get_datatransformer_object()
            target_column = "math score"
            numerical_columns = ["reading score","writing score"]

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info(
                "Applying Preprocessing Object on Training and Testing DataFrames."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f"Saved Processing Object...")

            save_object(
                file_path=self.data_transformation_confiq.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_confiq.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            