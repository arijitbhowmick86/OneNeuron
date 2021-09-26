# OneNeuron
Single Perceptron


## Add url  -- Visit readme.so website for more info
[Git Hub][https://github.com/arijitbhowmick86/OneNeuron]

## Add Image
![Image][]

## Main function
```python

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

```
