# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import math
import numpy as np
from scipy.sparse.csr import csr_matrix
from scipy.sparse import lil_matrix


def matrix_from_iterables_and_index_maps(iterables, item_to_index: dict, iterable_to_index: dict) -> csr_matrix:
    matrix = lil_matrix((len(item_to_index), len(iterable_to_index)), dtype='int')
    for iterable in iterables:
        for item in iterable:
            matrix[item_to_index[item], iterable_to_index[iterable]] += 1
    return matrix.tocsr()


def vector_from_index_and_value_maps(to_index: dict, to_value, length=None):
    if length is None:
        length = len(to_index)
    vector = zero_vector_from_length(length)
    for key, value in to_value.items():
        vector[to_index[key]] = value
    return vector


def dict_from_index_map_and_vector(to_index, vector):
    return {item: vector[index] for item, index in to_index.items()}


def count_nonzero_entries_in_matrix_row(matrix, row_index):
    row = matrix.getrow(row_index)
    return row.getnnz()


def cosine_distance(vector0, vector1):
    distance, _, _ = verbose_cosine_distance(vector0, vector1)
    return distance


def verbose_cosine_distance(vector0, vector1):
    normalized_vector0, norm0 = verbose_normalize(vector0)
    normalized_vector1, norm1 = verbose_normalize(vector1)
    return 1. - scalar_product(normalized_vector0, normalized_vector1), norm0, norm1


def scalar_product(vector0, vector1):
    return np.dot(vector0, vector1)


def normalize(vector):
    normalized_vector, _ = verbose_normalize(vector)
    return normalized_vector


def verbose_normalize(vector):
    if is_zero_vector(vector):
        return vector, 0
    vector_norm = norm(vector)
    return vector / vector_norm, vector_norm


def is_zero_vector(vector):
    return not np.any(vector)


def norm(vector):
    return math.sqrt(scalar_product(vector, vector))


def coefficient_wise_vector_product(vector0: np.ndarray, vector1: np.ndarray) -> np.ndarray:
    return np.multiply(vector0, vector1)


def matrix_vector_product(matrix: csr_matrix, vector: np.ndarray) -> np.ndarray:
    return matrix.dot(vector)


def dot_matrix_dot_products(dot_vector0, matrix, dot_vector1, vector):
    vector = coefficient_wise_vector_product(dot_vector1, vector)
    vector = matrix_vector_product(matrix, vector)
    vector = coefficient_wise_vector_product(dot_vector0, vector)
    return vector


def zero_vector_from_length(length: int) -> np.ndarray:
    return np.zeros(length)


def one_vector_from_length(length: int) -> np.ndarray:
    return np.ones(length)


def rescale_vector_to_satisfy_lower_negative_bound(vector, lower_bound):
    min_element = min(vector)
    if min_element < lower_bound:
        vector = lower_bound / min_element * vector
    return vector


def transpose_matrix(matrix):
    return matrix.transpose()


def create_vector(coefficients):
    return np.array(coefficients)


def are_equal_vectors(vector0, vector1):
    return np.array_equal(vector0, vector1)


def are_almost_equal_vectors(vector0, vector1):
    if len(vector0) != len(vector1):
        return False
    for index in range(len(vector0)):
        if not math.isclose(vector0[index], vector1[index]):
            return False
    return True


def are_almost_colinear_vectors(vector0, vector1):
    if is_zero_vector(vector0):
        return True
    for index in range(len(vector0)):
        if vector0[index] != 0.:
            return are_almost_equal_vectors(vector0 / vector0[index] * vector1[index], vector1)
