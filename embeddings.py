import numpy as np
from scipy import spatial

def load_n_glove_embeddings(n : int):
    count = 0
    embeddings_dict = {}
    with open("embeddings/glove.42B.300d.txt", 'r') as f:
        for line in f:
            if count >= n:
                break
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], "float32")
            embeddings_dict[word] = vector
            count += 1
    return embeddings_dict

def find_closest_embeddings(embeddings_dict, embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: spatial.distance.euclidean(embeddings_dict[word], embedding))

def continuous_to_discrete(embedding):
    new = np.zeros(100_000)
    for k in range(0, len(embedding)):
        if np.abs(embedding[k]) < 0.5:
            new[(300*k):(300*k+300)] = np.random.choice([1, 0], 300, p=[0.05, 0.95])
        if embedding[k] >= 0.5:
            new[(300*k):(300*k+300)] = np.concatenate((np.random.choice([1, 0], 150, p=[0.2, 0.8]),
                                                     np.random.choice([1, 0], 150, p=[0.05, 0.95])))
        if embedding[k] <= -0.5:
            new[(300*k):(300*k+300)] = np.concatenate((np.random.choice([1, 0], 150, p=[0.2, 0.8]),
                                                     np.random.choice([1, 0], 150, p=[0.05, 0.95])))
    return new