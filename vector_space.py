# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from matrix_operations import *


class VectorSpace:

    def __init__(self, iterables):
        self.item_to_index = map_to_index_from_iterable(iterables_union(iterables))
        self.iterable_to_index = map_to_index_from_iterable(iterables)
        self.item_iterable_matrix = matrix_from_iterables_and_index_maps(
            iterables, self.item_to_index, self.iterable_to_index)

    def item_vector_from_dict(self, item_distribution):
        return vector_from_index_and_value_maps(self.item_to_index, item_distribution)

    def iterable_vector_from_dict(self, iterable_distribution):
        return vector_from_index_and_value_maps(self.iterable_to_index, iterable_distribution)

    def item_dict_from_vector(self, item_vector):
        return dict_from_index_map_and_vector(self.item_to_index, item_vector)

    def iterable_dict_from_vector(self, iterable_vector):
        return dict_from_index_map_and_vector(self.iterable_to_index, iterable_vector)

    def iterable_vector_from_collection(self, iterable_collection):
        iterable_distribution = constant_distribution_from_collection(iterable_collection)
        return self.iterable_vector_from_dict(iterable_distribution)

    def count_iterables_containing_item(self, item):
        if item not in self.item_to_index:
            return 0
        return count_nonzero_entries_in_matrix_row(self.item_iterable_matrix, self.item_to_index[item])


def map_to_index_from_iterable(iterable):
    dictionary = dict()
    index = 0
    for item in iterable:
        if item not in dictionary:
            dictionary[item] = index
            index += 1
    return dictionary


def iterables_union(iterables):
    for iterable in iterables:
        for item in iterable:
            yield item


def constant_distribution_from_collection(collection):
    return {element: 1. for element in collection}
