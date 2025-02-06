from sklearn.neighbors import NearestNeighbors

def get_nearest_neighbors(train, amount):
    nbrs = NearestNeighbors(n_neighbors=amount, metric="euclidean", algorithm='brute').fit(train)
    I = nbrs.kneighbors(train, return_distance=False)
    return I