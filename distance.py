# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from matrix_operations import *
from vector_space import VectorSpace


class Distance(VectorSpace):

    def __init__(self, iterables, item_to_weight=None, iterable_to_weight=None):
        super().__init__(iterables)
        #
        self.item_weights_vector = None
        if item_to_weight is None:
            item_to_weight = self.tfidf_item_weights()
        self.set_item_weights(item_to_weight)
        #
        self.iterable_weights_vector = None
        if iterable_to_weight is None:
            iterable_to_weight = {iterable: 1 / len(iterable) for iterable in self.iterable_to_index}
        self.set_iterable_weights(iterable_to_weight)

    def __call__(self, iterables0, iterables1):
        distance, _, _, _, _, _, _ = self.verbose_distance(iterables0, iterables1)
        return distance

    def vectorize(self, iterables):
        vectorization, _ = self.verbose_vectorize(iterables)
        return vectorization

    def set_item_weights(self, item_to_weight):
        item_to_weight = normalize_distribution(item_to_weight)
        self.item_weights_vector = self.item_vector_from_dict(item_to_weight)

    def set_iterable_weights(self, iterable_to_weight):
        iterable_to_weight = normalize_distribution(iterable_to_weight)
        self.iterable_weights_vector = self.iterable_vector_from_dict(iterable_to_weight)

    def get_item_weights(self):
        return self.item_dict_from_vector(self.item_weights_vector)

    def get_iterable_weights(self):
        return self.iterable_dict_from_vector(self.iterable_weights_vector)

    def tfidf_item_weights(self):
        iterable_number = len(self.iterable_to_index)
        # We use log_of_ratio_zero_if_null_denominator to handle the case
        # where the only iterable containing an item has been removed (operation currently not supported).
        return {item: log_of_ratio_zero_if_null_denominator(iterable_number, self.count_iterables_containing_item(item))
                for item in self.item_to_index}

    def verbose_distance(self, iterables0, iterables1):
        vectorization0, iterables_vector0 = self.verbose_vectorize(iterables0)
        vectorization1, iterables_vector1 = self.verbose_vectorize(iterables1)
        distance, norm0, norm1 = verbose_cosine_distance(vectorization0, vectorization1)
        return distance, iterables_vector0, vectorization0, norm0, iterables_vector1, vectorization1, norm1

    def verbose_vectorize(self, iterables):
        iterables_vector = self.iterable_vector_from_collection(iterables)
        vectorization = dot_matrix_dot_products(self.item_weights_vector, self.item_iterable_matrix,
                                                self.iterable_weights_vector, iterables_vector)
        return vectorization, iterables_vector


def log_of_ratio_zero_if_null_denominator(numerator, denominator):
    if denominator == 0:
        return 0.
    return math.log(numerator / denominator)


def normalize_distribution(distribution):
    normalization_factor = sum(distribution.values())
    return {item: value / normalization_factor for item, value in distribution.items()}
