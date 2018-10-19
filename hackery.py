#!/usr/bin/env python

import argparse
import typing
import numpy as np
import numpy.testing as npt

def batch_knn(
    neighbors: int,
    needle_X: typing.Any,
    haystack_Xy: typing.Any,
    block_size: typing.Optional[int] = None
) -> typing.Tuple[np.array, np.array]:
    """
    Find y's corresponding to the nearest neighbors of needle_X.

    Input needle_X must have one fewer column than haystack_Xy.  All columns
    of needle_X and the all-but-last of haystack_Xy are coordinates.
    The last column of haystack_Xy contains the values There must be
    at least as many haystack rows as neighbors.  Result are the nearest
    (distances**2, values) to needle_X, where the rows correspond to needle_X
    rows and columns to neighbors.  Rows in the results are *not* sorted.
    Input needle_X is processed in stages when block_size is provided.
    """
    # Coerce to nicely strided data and check shape congruence
    needle_X = np.asfortranarray(needle_X)
    needle_len, _ = needle_X.shape
    haystack_Xy = np.asfortranarray(haystack_Xy)
    haystack_len, _ = haystack_Xy.shape
    assert needle_X.shape[1] + 1 == haystack_Xy.shape[1]
    assert 0 < neighbors <= haystack_len

    # When block_size is provided, subdivide needle_X.
    # Subdivision in this manner is over read-only, Fortran-ordered data.
    if block_size is not None:
        raise NotImplementedError()

    # Storage for the nearest distances and their corresponding values
    neighbors_d = np.empty((needle_len, neighbors), dtype=float, order='F')
    neighbors_y = np.empty_like(neighbors_d)

    # Fill the initial distances**2 from the top of the haystack
    scratch = np.empty_like(needle_X, dtype=float, order='F')
    for i in range(neighbors):
        np.subtract(needle_X, haystack_Xy[i, :-1], out=scratch)
        np.square(scratch, out=scratch)
        np.sum(scratch, axis=1, out=neighbors_d[:, i])

    # Fill the initial neighbors from the top of the haystack
    neighbors_y[:, :neighbors] = haystack_Xy[:neighbors, -1]

    # Process the remainder of the haystack...
    distance = np.empty((needle_len,), dtype=float, order='F')
    values = np.empty_like(distance)
    for i in range(neighbors, haystack_len):
        # ...by first computing distances**2.
        np.subtract(needle_X, haystack_Xy[i, :-1], out=scratch)
        np.square(scratch, out=scratch)
        np.sum(scratch, axis=1, out=distance)

        # ...by then 'bubbling away' distant data.  Yes, one bubble pass.
        np.copyto(dst=values, src=haystack_Xy[i, -1])
        for neighbors_dj, neighbors_yj in zip(neighbors_d.T, neighbors_y.T):
            mask = distance < neighbors_dj
            # Swap distance[mask], neighbors_d[mask, j] in fast manner.
            distance, neighbors_dj[:] = (
                    np.where(mask, neighbors_dj, distance),
                    np.where(mask, distance, neighbors_dj))
            # Swap values[mask], neighbors_y[mask, j] in fast manner.
            values, neighbors_yj[:] = (
                    np.where(mask, neighbors_yj, values),
                    np.where(mask, values, neighbors_yj))

    # Return the k-nearest (distances**2, values).
    return neighbors_d, neighbors_y


def test_1neighbor_1d():
    needle_X = np.array([[0.1],
                         [0.9],
                         [2.1],
                         [2.9],
                         [4.1]])
    haystack_Xy = np.array([[0., 0.],
                            [1., 1.],
                            [2., 2.],
                            [3., 3.],
                            [4., 4.]])
    d, y = batch_knn(1, needle_X, haystack_Xy)
    npt.assert_array_equal(y, haystack_Xy[:, 1:])
    npt.assert_almost_equal(d, 0.01 * np.ones((5, 1)))


def test_2neighbor_1d():
    needle_X = np.array([[0.1],
                         [0.9],
                         [2.1],
                         [2.9],
                         [4.1]])
    haystack_Xy = np.array([[0., 0.],
                            [1., 1.],
                            [2., 2.],
                            [3., 3.],
                            [4., 4.]])
    d, y = batch_knn(2, needle_X, haystack_Xy)
    npt.assert_array_equal(y, np.array([[0.0, 1.0],
                                        [0.0, 1.0],
                                        [2.0, 3.0],
                                        [3.0, 2.0],
                                        [4.0, 3.0]]))
    npt.assert_almost_equal(d, np.array([[0.01, 0.81],
                                         [0.81, 0.01],
                                         [0.01, 0.81],
                                         [0.01, 0.81],
                                         [0.01, 1.21]]))


def test_1neighbor_2d():
    needle_X = np.array([[0.1, 0.0],
                         [0.9, 1.0],
                         [2.1, 2.0],
                         [2.9, 3.0],
                         [4.1, 4.0]])
    haystack_Xy = np.array([[0., 0., 0.],
                            [1., 1., 1.],
                            [2., 2., 2.],
                            [3., 3., 3.],
                            [4., 4., 4.]])
    d, y = batch_knn(1, needle_X, haystack_Xy)
    npt.assert_array_equal(y, haystack_Xy[:, 2:])
    npt.assert_almost_equal(d, 0.01 * np.ones((5, 1)))


def test_3neighbor_2d():
    needle_X = np.array([[0.0, 0.0]])
    haystack_Xy = np.array([[2., 9., 3.],  # Ensure pushed out
                            [3., 9., 4.],
                            [4., 8., 5.],
                            [5., 7., 6.],
                            [0., 0., 0.],  # Desired starts here
                            [1., 5., 2.],
                            [1., 1., 1.]])
    d, y = batch_knn(3, needle_X, haystack_Xy)
    assert (y < 3).all()
    assert (0**2 + 0**2) in d
    assert (1**2 + 1**2) in d
    assert (1**2 + 5**2) in d


# Very simple driver for performance testing
if __name__ == '__main__':
    # Parse incoming arguments
    p = argparse.ArgumentParser()
    p.add_argument('seed',      type=int, help='Random seed')
    p.add_argument('dimension', type=int, help='# of coordinates')
    p.add_argument('neighbors', type=int, help='# of neighbors to find per needle')
    p.add_argument('needles',   type=int, help='# of vectors to find')
    p.add_argument('haystacks', type=int, help='# of vectors to search')
    args = p.parse_args()

    # Generate the sample data
    random      = np.random.RandomState(args.seed)
    needle_X    = random.randn(args.needles, args.dimension)
    haystack_Xy = random.randn(args.haystacks, args.dimension + 1)
    d, y = batch_knn(args.neighbors, needle_X, haystack_Xy)
