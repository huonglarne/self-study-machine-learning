import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import distance

data = np.genfromtxt("seeds_dataset.txt")
test_index = random.sample(range(len(data)), 15) # take random samples from the data set
print(test_index)
k=10 # number of closest neighbors

result = []

for i in test_index:
    test_sample = data[i]
    train_data = np.concatenate((data[:i],data[i+1:]), axis=0)

    # calculate euclidean distance from the sample to all points in the training data set
    distances = [[distance.euclidean(test_sample[:-1], x[:-1]), x[-1]] for x in train_data]
    distances.sort()
    distances=np.asarray(distances)[:10, -1]

    # select the majority of the labels of top k neighbors and compare with the test sample's label
    top_k = round(np.mean(distances))
    if test_sample[-1] == top_k:
        result.append(True)
    else:
        result.append(False)

print(np.mean(result)) # the accuracy usually varies from 0.8 to 0.9