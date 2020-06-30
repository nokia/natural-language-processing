# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
from matrix_operations import *


iterables = ['banana', 'ananas', 'base']
item_to_index = {'b': 0, 'a': 1, 'n': 2, 's': 3, 'e': 4}
iterable_to_index = {'banana': 0, 'ananas': 1, 'base': 2}
matrix = csr_matrix([[1, 0, 1], [3, 3, 1], [2, 2, 0], [0, 1, 1], [0, 0, 1]])


class TestMatrixOperations(unittest.TestCase):

    def test_matrix_from_iterables_and_index_maps(self):
        computed = matrix_from_iterables_and_index_maps(iterables, item_to_index, iterable_to_index)
        expected = matrix
        row_number, column_number = computed.get_shape()
        self.assertEqual((row_number, column_number), (5, 3), 'wrong shape')
        for i in range(row_number):
            for j in range(column_number):
                self.assertEqual(computed[i, j], expected[i, j])

    def test_vector_from_index_and_value_maps(self):
        to_index = {'a': 0, 'b': 1, 'c': 2}
        to_value = {'a': 0.1, 'c': 3}
        computed = vector_from_index_and_value_maps(to_index, to_value)
        expected = [0.1, 0., 3.]
        for i in range(len(to_index)):
            self.assertEqual(computed[i], expected[i])

    def test_count_nonzero_entries_in_matrix_row(self):
        row_number, _ = matrix.get_shape()
        self.assertEqual(row_number, 5)
        computed = [count_nonzero_entries_in_matrix_row(matrix, i) for i in range(row_number)]
        expected = [2, 3, 2, 2, 1]
        for i in range(row_number):
            self.assertEqual(computed[i], expected[i])

    def test_cosine_distance(self):
        u = create_vector([1., 3., 2.])
        v = create_vector([2., -1., 0.5])
        zero_vector = create_vector([0., 0., 0.])
        self.assertAlmostEqual(cosine_distance(zero_vector, u), 1.)
        self.assertAlmostEqual(cosine_distance(u, v), 1. - scalar_product(u, v) / norm(u) / norm(v))

    def test_scale_vector_to_satisfy_lower_bound(self):
        vector = create_vector([6, 2, 4, 8])
        self.assertTrue(are_equal_vectors(rescale_vector_to_satisfy_lower_negative_bound(vector, -1), vector))
        vector = create_vector([6, -2, 4, 8])
        self.assertTrue(are_equal_vectors(rescale_vector_to_satisfy_lower_negative_bound(vector, -1),
                                          create_vector([3, -1, 2, 4])))


if __name__ == '__main__':
    unittest.main()
