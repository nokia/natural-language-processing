# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
from distance import *

iterables = ['aa', 'ab', 'bbb']
item_to_weight = {'a': 1, 'b': 2}
iterable_to_weight = {'aa': 1, 'ab': 2, 'bbb': 3}
distance = Distance(iterables, item_to_weight, iterable_to_weight)
iterables0 = {'ab'}
iterables1 = {'bbb'}


class MyTestCase(unittest.TestCase):

    def test_vectorize(self):
        vector0 = distance.vectorize(iterables0)
        vector1 = distance.vectorize(iterables1)
        self.assertTrue(are_almost_colinear_vectors(vector0, create_vector([2, 4])) or
                        are_almost_colinear_vectors(vector0, create_vector([4, 2])))
        self.assertTrue(are_almost_colinear_vectors(vector1, create_vector([0, 18])) or
                        are_almost_colinear_vectors(vector1, create_vector([18, 0])))
        self.assertTrue(are_almost_colinear_vectors(vector0 + vector1, distance.vectorize({'ab', 'bbb'})))

    def test_tfidf(self):
        tfidf_distance = Distance(iterables)
        expected_item_weights_vector = create_vector([math.log(3/2), math.log(3/2)])
        expected_item_weights_vector /= sum(expected_item_weights_vector)
        self.assertTrue(are_equal_vectors(tfidf_distance.item_weights_vector, expected_item_weights_vector))

    def test_verbose_distance(self):
        d, iv0, vz0, n0, iv1, vz1, n1 = distance.verbose_distance(iterables0, iterables1)
        self.assertAlmostEqual(d, 1. - scalar_product(vz0, vz1) / n0 / n1)
        self.assertTrue(are_equal_vectors(dot_matrix_dot_products(distance.item_weights_vector,
                                                                  distance.item_iterable_matrix,
                                                                  distance.iterable_weights_vector, iv0),
                                          vz0))
        self.assertTrue(are_equal_vectors(dot_matrix_dot_products(distance.item_weights_vector,
                                                                  distance.item_iterable_matrix,
                                                                  distance.iterable_weights_vector, iv1),
                                          vz1))


if __name__ == '__main__':
    unittest.main()
