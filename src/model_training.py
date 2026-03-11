# 1.load processed data from processed folder
# 2.create model and train the model
# 3.save the model in artifacts folder

import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(base_dir, "data", "processed")
artifacts_dir = os.path.join(base_dir, "artifacts")
x_train = pd.read_csv(os.path.join(processed_dir, "x_train.csv"))
y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))
x_test = pd.read_csv(os.path.join(processed_dir, "x_test.csv"))
y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))

print(x_train)

model=LinearRegression()
model.fit(x_train,y_train)

os.makedirs(artifacts_dir, exist_ok=True)
with open(os.path.join(artifacts_dir, "model.pkl"), "wb") as f:
    pickle.dump(model, f)