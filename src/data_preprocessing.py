# 1.load raw data
# 2.identify x and y(input and output)
# 3.split data into train and test

import pandas as pd
from sklearn.model_selection import train_test_split
def load_split_data():
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "raw", "insurance_data.csv")
    df = pd.read_csv(data_path)
    x=df[["Age","Annual_Income_LPA","Policy_Term_Years","Sum_Assured_Lakhs"]]
    y=df["Annual_Premium_Thousands"]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test
print("successfully loaded and split the data")