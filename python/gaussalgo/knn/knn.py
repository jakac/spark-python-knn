from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from pyspark.mllib.linalg import Vector

def compute_neighbors(X, Y, top_n=10, metric=lambda x,y: cosine_similarity(x,y)[0,0], is_similarity=True,
                          X_blocks=None, Y_blocks=None):
        """
        Computes for each vector from X top_n nearest neighbors from Y using specified metric
        :param X: RDD of (id, vector) vector can be anything which is passed to metric function
        :param Y: RDD of (id, vector) vector can be anything which is passed to metric function
        :param top_n: int How many neighbors
        :param metric: string or callable The metric to use when calculating distance between instances. If metric is a string, it must be one of the options allowed by scipy.spatial.distance.pdist for its metric parameter, or a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        :param is_similarity: bool Whether metric is similarity or distance
        :param X_blocks: int Repartition X to X_blocks.
        :param Y_blocks: int Repartition Y to Y_blocks.
        :return: RDD of (left_id, [(right_id, score)]) sorted by similarity|distance
        """
        # Repartition
        if X_blocks is not None:
            X = X.repartition(X_blocks)
        if Y_blocks is not None:
            Y = Y.repartition(Y_blocks)

        if isinstance(metric, str):
            metric = lambda x, y: pairwise_distances(x, y, metric=metric)[0, 0]

        def blockify(X):
            def to_arrays(part):
                keys = []
                vecs = []
                for k, v in part:
                    keys.append(k)
                    vecs.append(v)
                if keys:
                    yield (keys, vecs)
            return X.mapPartitions(to_arrays)

        # Each partition to one-row block
        X = blockify(X)
        Y = blockify(Y)

        def process_block(row):
            """row = ((ids, vectors), (ids, vectors))"""
            # Iterate over rows of left matrix
            for l in range(len(row[0][1])):
                # Remember top n rows from right
                top = []
                if is_similarity:
                    worst = float('-inf')
                else:
                    worst = float('inf')
                left = row[0][1][l]
                # If vectors are Spark Vectors, convert to numpy.ndarray
                if isinstance(left, Vector):
                    left = left.toArray()
                # Iterate over rows of right matrix
                for r in range(len(row[1][1])):
                    # Compute distance/similarity
                    right = row[1][1][r]
                    # If vectors are Spark Vectors, convert to numpy.ndarray
                    if isinstance(right, Vector):
                        right = right.toArray()
                    dist = metric(left, right)
                    if len(top) < top_n or (is_similarity and dist > worst) or (not is_similarity and dist < worst):
                        # We found one of top, add to top list
                        if len(top) == 0:
                            top.append((row[1][0][r], dist))
                        else:
                            # Insert on right position in top list (we have sorted list)
                            for k in range(len(top)):
                                if (is_similarity and dist > top[k][1]) or \
                                    (not is_similarity and dist < top[k][1]):
                                    top.insert(k, (row[1][0][r], dist))
                                    break
                                if k == len(top) - 1 and len(top) < top_n:
                                    top.append((row[1][0][r], dist))
                            if len(top) > top_n:
                                del top[top_n]
                    worst = top[-1][1]
                yield (row[0][0][l], top)

        def reducer(top1, top2, k):
            li = top1
            li.extend(top2)
            return sorted(li, key=lambda x: x[1], reverse=is_similarity)[:k]


        return X.cartesian(Y).flatMap(process_block).foldByKey([], lambda top1, top2: reducer(top1, top2, top_n))

