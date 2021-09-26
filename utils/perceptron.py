import numpy as np
import pandas as pd
import logging
import os
from tqdm import tqdm


class Perceptron():

    def __init__(self, eta, epochs):
        
        np.random.seed(100)
        self.weights = np.random.randn(3) * 1e-4
        self.eta = eta
        self.epochs = epochs

    def activationFunction(self, data):
        return np.where(data > 0, 1, 0)

    def summationFunction(self, data, weights):
        return np.dot(data, weights)

    def fit(self, input):
        
        self.data = pd.DataFrame(input["data"])
        self.target = input["target"]
        
        bias = np.ones((len(self.data), 1))
        logging.info(f"bias : {bias}")
        logging.info(f"self.data : {self.data}")
        logging.info(f"self.target : {self.target}")
        logging.info(f"self.weights : {self.weights}, {self.weights.shape}")
        data_with_bias = np.c_[self.data, bias]
        logging.info(f"data_with_bias : {data_with_bias}")
        

        summation_output = self.summationFunction(data_with_bias, self.weights)
        logging.info(f"summation_output : {summation_output}")
        pred = self.activationFunction(summation_output)
        logging.info(f"Epoch : Random ,  Pred : {pred}")

        for epoch in tqdm(range(self.epochs), total=self.epochs, desc="training model"):
            
            self.error = self.target - pred
            if(np.all(self.error == 0)):
                logging.info(f"Target achieved in Epoch : {epoch - 1}")
                break
            else:               
                self.weights = self.weights + (self.eta * np.dot(data_with_bias.T, self.error))

                logging.info(f"weights : {self.weights}")
                pred = self.activationFunction(self.summationFunction(data_with_bias, self.weights))
                logging.info(f"Epoch : {epoch} ,  Pred : {pred}")

    def predict(self, data):
        bias = np.ones((len(data), 1))
        data_with_bias = np.c_[data, bias]
        return self.activationFunction(self.summationFunction(data_with_bias, self.weights))




