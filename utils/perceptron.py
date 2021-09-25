import numpy as np
import pandas as pd



class Perceptron():

    def __init__(self, eta, epochs):
        
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
        print(f"bias : {bias}")
        print(f"self.data : {self.data}")
        print(f"self.target : {self.target}")
        print(f"self.weights : {self.weights}, {self.weights.shape}")
        data_with_bias = np.c_[self.data, bias]
        print(f"data_with_bias : {data_with_bias}")
        

        summation_output = self.summationFunction(data_with_bias, self.weights)
        print(f"summation_output : {summation_output}")
        pred = self.activationFunction(summation_output)
        print(f"Epoch : Random ,  Pred : {pred}")

        for epoch in range(self.epochs):
            
            self.error = self.target - pred
            if(np.all(self.error == 0)):
                print(f"Target achieved in Epoch : {epoch - 1}")
                break
            else:               
                self.weights = self.weights + (self.eta * np.dot(data_with_bias.T, self.error))

                print(f"weights : {self.weights}")
                pred = self.activationFunction(self.summationFunction(data_with_bias, self.weights))
                print(f"Epoch : {epoch} ,  Pred : {pred}")

    def predict(self, data):
        bias = np.ones((len(data), 1))
        data_with_bias = np.c_[data, bias]
        return self.activationFunction(self.summationFunction(data_with_bias, self.weights))


input = {
    "data" : {
        "x1" : [0, 0, 1, 1],
        "x2" : [0, 1, 0, 1]
    },
    "target" : [0, 0, 0, 1]
}
np.random.seed(100)
weights = np.random.rand(3) * 1e-4
eta = 0.1
epochs = 20
p = Perceptron(eta, epochs)
p.fit(input)
print(p.predict(pd.DataFrame(input["data"])))