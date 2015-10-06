# Spark Python K-nn
Simple but memory efficient function for computation of K nearest neighbors.

## Requires
Python 2.7. Installs Numpy, scikit-learn.

## Install
Add gaussalgo/python-knn:1.0.0 to requirements of your app:

    $SPARK_HOME/bin/pyspark --packages gaussalgo:python-knn:1.0.0
    
## Usage

    ```python
    from gaussalgo.knn import compute_neighbors
    import numpy as np
    
    left = sc.parallelize([
            (1, np.array([0,0,1,1])),
            (2, np.array([0,1,1,1])),
            (3, np.array([0,0,1,1])),
            (4, np.array([1,1,1,1]))
    ])
    right = sc.parallelize([
            (5, np.array([0,0,1,1])),
            (6, np.array([0,1,1,1])),
            (7, np.array([1,0,0,1])),
            (8, np.array([1,1,1,1]))
    ])

    neighbors = compute_neighbors(left, right, top_n=2).collect()
    returns: [(1, [(5, 0.99999999999999978), (6, 0.81649658092772603)]), (2, [(6, 1.0000000000000002), (8, 0.86602540378443882)]), (3, [(5, 0.99999999999999978), (6, 0.81649658092772603)]), (4, [(8, 1.0), (6, 0.86602540378443882)])]
    ```

Options:
    * top_n: int How many neighbors compute
    * metric: callable which for two vectors return their distance/similarity. If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist for its metric parameter, or a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
    * is_similarity: boolean If metric is similarity or distance
    * X_blocks, Y_block: int Repartition X or Y to specified number of partitions.

## Tips
Computational complexity is still $X \times Y$, it uses brute force for K-nn search, but it do not need $X \times Y$ of memory.
Memory usage is X_PARTITION_SIZE $\times$ NUM_X_PARTITIONS $\times$ Y_PARTITION_SIZE $\times$ NUM_Y_PARTITIONS, so choosing right number of partitions is crucial.

## Contribution
Feel free to comment it or contribute to it.