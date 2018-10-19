#!/usr/bin/env python

import typing
import numpy as np
import numpy.testing as npt

def batch_knn(
    neighbors: int,
    needle_X: typing.Any,
    haystack_Xy: typing.Any,
) -> typing.Tuple[np.array, np.array]:
    """
    Find y's corresponding to the nearest neighbors of needle_X.

    Input needle_X must have one fewer column than haystack_Xy.  All columns
    of needle_X and the all-but-last are coordinates.  The last column
    of haystack_Xy contains the values There must be at least as many
    haystack rows as neighbors.  Result are the nearest (distances, values)
    to needle_X, where the rows correspond to needle_X rows and columns
    to neighbors.
    """
    # Coerce to nicely strided data and check shape congruence
    needle_X = np.asfortranarray(needle_X)
    needle_len, _ = needle_X.shape
    haystack_Xy = np.asfortranarray(haystack_Xy)
    haystack_len, _ = haystack_Xy.shape
    assert needle_X.shape[1] + 1 == haystack_Xy.shape[1]
    assert 0 < neighbors <= haystack_len

    # Storage for the nearest neighbors and their corresponding value
    neighbors_d = np.empty((needle_len, neighbors), dtype=float, order='F')
    neighbors_y = np.empty_like(neighbors_d)

    # Fill the initial distances from the top of the haystack
    scratch = np.empty_like(needle_X, dtype=float)
    for i in range(neighbors):
        np.subtract(needle_X, haystack_Xy[i, :-1], out=scratch)
        np.square(scratch, out=scratch)
        np.sum(scratch, axis=1, out=neighbors_d[:, i])

    # Fill the initial neighbors from the top of the haystack
    neighbors_y[:, :neighbors] = haystack_Xy[:neighbors, -1]

    # Process the remainder of the haystack...
    distance = np.empty((needle_len,), dtype=float)
    values = np.empty_like(distance)
    mask = np.empty((needle_len,), dtype=bool)
    for i in range(neighbors, haystack_len):
        # ...by first computing distances.
        np.subtract(needle_X, haystack_Xy[i, :-1], out=scratch)
        np.square(scratch, out=scratch)
        np.sum(scratch, axis=1, out=distance)

        # ...by then 'bubbling away' further data.
        # TODO Remove temporaries in swap operations.
        np.copyto(dst=values, src=haystack_Xy[i, -1], casting='same_kind')
        for j in range(neighbors):
            np.less(distance, neighbors_d[:, j], out=mask)
            distance[mask], neighbors_d[mask, j] = neighbors_d[mask, j], distance[mask]
            values[mask], neighbors_y[mask, j] = neighbors_y[mask, j], values[mask]

    # Return the k-nearest (distances, values).
    return neighbors_d, neighbors_y


def test_1neighbor_1d():
    needle_X=np.array([[0.1],
                       [0.9],
                       [2.1],
                       [2.9],
                       [4.1]])
    haystack_Xy=np.array([[0., 0.],
                          [1., 1.],
                          [2., 2.],
                          [3., 3.],
                          [4., 4.]])
    d, y = batch_knn(1, needle_X, haystack_Xy)
    npt.assert_array_equal(y, haystack_Xy[:, 1:])
    npt.assert_almost_equal(d, 0.01 * np.ones((5, 1)))


def test_1neighbor_2d():
    needle_X=np.array([[0.1, 0.0],
                       [0.9, 1.0],
                       [2.1, 2.0],
                       [2.9, 3.0],
                       [4.1, 4.0]])
    haystack_Xy=np.array([[0., 0., 0.],
                          [1., 1., 1.],
                          [2., 2., 2.],
                          [3., 3., 3.],
                          [4., 4., 4.]])
    d, y = batch_knn(1, needle_X, haystack_Xy)
    npt.assert_array_equal(y, haystack_Xy[:, 2:])
    npt.assert_almost_equal(d, 0.01 * np.ones((5, 1)))

# def test_thing():
#     needle_X = np.random.randn(100, 2)
#     haystack_Xy = np.random.randn(1000, 3)
#     result = batch_knn(needle_X, haystack_Xy, 5)
#     assert False
