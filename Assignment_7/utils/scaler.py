from sklearn import preprocessing
import pandas as pd

def scale(dataset):
    # Perform the sklearn scaler on a pandas dataframe
    dataset = preprocessing.scale(dataset)
    # Turn into a pandas df again
    d = {'x'+str(i+1) : dataset[:,i] for i in range(len(dataset[0]))}
    dataset = pd.DataFrame(d)
    return dataset
