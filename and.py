
from utils.perceptron import Perceptron
import pandas as pd
from utils.utils import save_model, load_model, set_up_logger
import logging
import os

def main():
    
    set_up_logger()

    logging.info("In main method")
    input = {
    "data" : {
        "x1" : [0, 0, 1, 1],
        "x2" : [0, 1, 0, 1]
    },
    "target" : [0, 0, 0, 1]
    }

    eta = 0.001
    epochs = 20
    p = Perceptron(eta=eta, epochs=epochs)
    p.fit(input)
    print(p.predict(pd.DataFrame(input["data"])))

    save_model(model=p, filename="and.model", dirname="resources")
    p_loaded = load_model(filename="and.model", dirname="resources")
    print(p.predict([[1, 1], [0, 1]]))

    logging.info("Exiting main method")

if __name__ == '__main__':
    main()