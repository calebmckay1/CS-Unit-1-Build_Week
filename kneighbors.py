import numpy as np


class Knearest:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def euclidean_distance(self, data1, data2):
        distance = 0
        for i in range(len(data1)):
            distance += (data1[i] - data2[i]) ** 2
            euclidian_distance = np.sqrt(distance)
        return euclidian_distance
    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        print('Training Complete.')
        
    def predict(self, X_test):
        # start predictions as empty list
        predictions = []
        
        # loop through length of X_test
        for i in range(len(X_test)):
            distances = []
            
            # every row in 'X_train' find the euclidean distance to 
            # X_test and append the distances list
            for row in self.X_train:
                d = self.euclidean_distance(row, X_test[i])
                distances.append(d)
                
            # sort the list and keep the neighbors stated in __init__
            neighbors = np.array(distances).argsort()[: self.n_neighbors]
            
            count = {}            
            for v in neighbors:
                if self.y_train[v] in count:
                    count[self.y_train[v]] += 1
                else:
                    count[self.y_train[v]] = 1
            predictions.append(max(count, key=count.get))
            
        return predictions

