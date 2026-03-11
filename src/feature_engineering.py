# 1.load training and testing data
# 2.scale the training data
# 3.save scaled data into processed folder

from data_preprocessing import load_split_data
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle


x_train,x_test,y_train,y_test=load_split_data()

# Ensure processed and artifacts directories exist
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
processed_dir = os.path.join(base_dir, "data", "processed")
artifacts_dir = os.path.join(base_dir, "artifacts")
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(artifacts_dir, exist_ok=True)

scaler=StandardScaler()

x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

pd.DataFrame(x_train_scaled).to_csv(os.path.join(processed_dir, "x_train.csv"), index=False)
pd.DataFrame(x_test_scaled).to_csv(os.path.join(processed_dir, "x_test.csv"), index=False)
pd.DataFrame(y_train).to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
pd.DataFrame(y_test).to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

with open(os.path.join(artifacts_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("successfully scaled the data and saved the scaler object")