# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
from vector_space import *


iterables = ['banana', 'ananas', 'base']
vector_space = VectorSpace(iterables)
# item_to_index = {'b': 4, 'a': 3, 'n': 2, 's': 1, 'e': 0}
# iterable_to_index = {'banana': 1, 'ananas': 2, 'base': 0}
# matrix = csc_matrix([[0, 0, 1],
#                      [0, 1, 1],
#                      [2, 2, 0],
#                      [3, 3, 1],
#                      [1, 0, 1]])


class TestVectorSpace(unittest.TestCase):

    def test_map_to_index_from_iterable(self):
        iterable = 'abacbde'
        computed = map_to_index_from_iterable(iterable)
        computed_list = ['' for _ in range(len(iterable))]
        for item, index in computed.items():
            computed_list[index] = item
        for letter in iterable:
            self.assertIn(letter, computed_list)

    def test_iterables_union(self):
        string_list = ['', 'abc', 'de', '', 'f', '']
        computed = iterables_union(string_list)
        computed_string = ''
        for letter in computed:
            computed_string += letter
        self.assertEqual(computed_string, 'abcdef')

    def test_count_iterables_containing_item(self):
        self.assertEqual(vector_space.count_iterables_containing_item('a'), 3)
        self.assertEqual(vector_space.count_iterables_containing_item('b'), 2)
        self.assertEqual(vector_space.count_iterables_containing_item('n'), 2)
        self.assertEqual(vector_space.count_iterables_containing_item('e'), 1)
        self.assertEqual(vector_space.count_iterables_containing_item('f'), 0)

    def test_vector_length(self):
        vector = vector_space.iterable_vector_from_collection(['ananas', 'banana'])
        projection = matrix_vector_product(vector_space.item_iterable_matrix, vector)
        self.assertEqual(len(projection), len(vector_space.item_to_index))


if __name__ == '__main__':
    unittest.main()
