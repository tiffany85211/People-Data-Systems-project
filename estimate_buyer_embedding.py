import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors

N_NEIGHBORS = 5
DATA_FILEPATH = './data/'
MODEL_FILEPATH = './machine_learning_model/'

def openpickle(filename):
    with open(DATA_FILEPATH + filename, "rb") as readfile:
        loaded = pickle.load(readfile)
    return loaded

def dumppickle(var, filename):
    pickle.dump(var, open(MODEL_FILEPATH + filename, 'wb'))
    

def main():
    # load files
    train_covariate = openpickle('train_covariate')
    train_noisy_embedding = openpickle('train_noisy_embedding')
    test_covariate = openpickle('test_covariate')
    test_noisy_embedding = openpickle('test_noisy_embedding')

    existing_train_idx = list(train_noisy_embedding.index)
    existing_test_idx = list(test_noisy_embedding.index)

    existing_covariate = pd.concat([train_covariate.loc[existing_train_idx], test_covariate.loc[existing_test_idx]])

    train_noisy_embedding.columns = train_noisy_embedding.columns.astype(str)
    existing_embedding = pd.concat([train_noisy_embedding, test_noisy_embedding], axis=0)

    # train KNN to get new user vectors
    neigh = NearestNeighbors(n_neighbors = N_NEIGHBORS)
    neigh.fit(existing_covariate)

    #pickle the object and store it in a file
    existing_embedding.to_pickle(MODEL_FILEPATH + 'existing_embedding')
    dumppickle(neigh, 'knn_model')
  
if __name__=="__main__":
    main()

