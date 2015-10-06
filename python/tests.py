# You may include your unit tests in this file.

from pyspark import SparkContext
from gaussalgo.knn import compute_neighbors
import numpy as np

def init_sc():
    return SparkContext(appName='Python K-nn tests')

def test_np_vectors():
    sc = init_sc()
    left = sc.union([
        sc.parallelize([
            (1, np.array([0,0,1,1])),
            (2, np.array([0,1,1,1]))]),
        sc.parallelize([
            (3, np.array([0,0,1,1])),
            (4, np.array([1,1,1,1]))
    ])])
    right = sc.union([
        sc.parallelize([
            (5, np.array([0,0,1,1])),
            (6, np.array([0,1,1,1]))]),
        sc.parallelize([
            (7, np.array([1,0,0,1])),
            (8, np.array([1,1,1,1]))
    ])])

    neighbors = compute_neighbors(left, right, top_n=2).collect()
    one = [x for x in neighbors if x[0] == 1][0]
    print neighbors
    assert one[1][0][0] == 5 and one[1][1][0] == 6


if __name__ == '__main__':
    test_np_vectors()
