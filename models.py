import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import logging
import json

# Function definitions
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data.dropna()

def prepare_data(data, target, features, test_size, random_state):
    X = data[features]
    y = data[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train, model_type):
    model = model_type()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred, squared=False)

def load_config(config_path):
    with open(config_path) as config_file:
        return json.load(config_file)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Class definition
class MyModel:

    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.data = None
        self.model = None

    def load_data(self):
        self.data = load_data(self.config["data_path"])

    def train(self):
        try:
            X_train, X_test, y_train, y_test = prepare_data(self.data, self.config["target"], self.config["features"], self.config["test_size"], self.config["random_state"])
            self.model = train_model(X_train, y_train, self.config["model_type"])
            logging.info("Model training completed.")
        except Exception as e:
            logging.error("Exception occurred", exc_info=True)

    def evaluate(self):
        try:
            X_train, X_test, y_train, y_test = prepare_data(self.data, self.config["target"], self.config["features"], self.config["test_size"], self.config["random_state"])
            return evaluate_model(self.model, X_test, y_test)
        except Exception as e:
            logging.error("Exception occurred", exc_info=True)

    def save_model(self):
        joblib.dump(self.model, self.config["model_path"])
        logging.info("Model saved.")

    def load_model(self):
        self.model = joblib.load(self.config["model_path"])
        logging.info("Model loaded.")

    def predict(self, input_data):
        try:
            return self.model.predict(input_data)
        except Exception as e:
            logging.error("Exception occurred", exc_info=True)

# You can now create an object of MyModel and use it.
model = MyModel('config.json')
model.load_data()
model.train()
print(model.evaluate())
model.save_model()

# Load the model and predict
loaded_model = MyModel('config.json')
loaded_model.load_model()
print(loaded_model.predict([[1.2, 3.4, 5.6]]))  # replace with your actual input data
