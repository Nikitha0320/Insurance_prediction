# 1.load scaler file and model.pkl file
# 2.create a function to predict
import pickle
import numpy as np

class Insurance_Prediction:
    def __init__(self):
        with open("C:\\tekworks\\Projects\\Insurance_prediction\\artifacts\\scaler.pkl","rb") as f:
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            scaler_path = os.path.join(base_dir, "artifacts", "scaler.pkl")
            self.scaler = pickle.load(f)
        with open("C:\\tekworks\\Projects\\Insurance_prediction\\artifacts\\model.pkl","rb") as f:
            model_path = os.path.join(base_dir, "artifacts", "model.pkl")
            self.model = pickle.load(f)
    def prediction(self,Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs):
        input=np.array([[Age,Annual_Income_LPA,Policy_Term_Years,Sum_Assured_Lakhs]])
        scaled_input=self.scaler.transform(input)
        result=self.model.predict(scaled_input)
        return result[0]