# 1.load scaler file and model.pkl file
# 2.create a function to predict
import pickle
import numpy as np

import os

class Insurance_Prediction:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        artifacts_path = os.path.join(base_path, "..", "artifacts")

        scaler_file = os.path.join(artifacts_path, "scaler.pkl")
        with open(scaler_file, "rb") as f:
            self.scaler = pickle.load(f)

        model_file = os.path.join(artifacts_path, "model.pkl")
        with open(model_file, "rb") as f:
            self.model = pickle.load(f)

    def prediction(self, Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs):
        input = np.array([[Age, Annual_Income_LPA, Policy_Term_Years, Sum_Assured_Lakhs]])
        scaled_input = self.scaler.transform(input)
        result = self.model.predict(scaled_input)
        return result[0]
