import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
print np.array(zip(iris.data,iris.target))[0:10]
print(iris.DESCR)
