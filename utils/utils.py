
import joblib
import os
import logging



def set_up_logger():
    logging_format = "[ %(asctime)s : %(levelname)s : %(module)s ] : %(message)s"
    os.makedirs("logs", exist_ok=True)
    logger_path = os.path.join("logs", "logger.log")

    logging.basicConfig(filename=logger_path, level=logging.INFO, format=logging_format,filemode="a")


def save_model(model, filename, dirname):
    logging.info("Entering save model")
    path = os.path.join(dirname, filename)
    joblib.dump(value=model, filename=path)
    logging.info("Exiting save model")

def load_model(filename, dirname):
    logging.info("Entering load model")
    path = os.path.join(dirname, filename)
    model = joblib.load(path)
    logging.info("Exiting load model")
    return model