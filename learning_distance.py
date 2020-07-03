# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import random
from matrix_operations import *
from distance import Distance


DEFAULT_NUMBER_OF_ITERATIONS = 5


class LearningDistance(Distance):

    def __init__(self, iterables, item_to_weight=None, iterable_to_weight=None):
        super().__init__(iterables, item_to_weight, iterable_to_weight)

    def learn(self, oracle_claims, ratio_item_iterable_learning=0.5, convergence_speed=0.5,
              number_of_iterations=DEFAULT_NUMBER_OF_ITERATIONS):
        oracle_claims = list(oracle_claims)
        for _ in range(number_of_iterations):
            random.shuffle(oracle_claims)
            for oracle_claim in oracle_claims:
                self.learn_from_one_oracle_claim(oracle_claim,
                                                 ratio_item_iterable_learning=ratio_item_iterable_learning,
                                                 effort=convergence_speed)

    def learn_from_one_oracle_claim(self, oracle_claim, ratio_item_iterable_learning=0.5, effort=1.):
        """ 'effort' is a value between '0.' and '1.'. It represents the amplitude of the change applied to the weights
        so that the distance conforms to 'oracle_claim'.
        Let 't' denote the target distance between the two sets of iterables from oracle_claim,
        'c' their current distance, and 'd' the distance achieved after the update.
        Then 'effort' is around '(t - d) / (t - c)'. """
        enriched_oracle_claim = EnrichedOracleClaim(oracle_claim, self, effort=effort)
        if enriched_oracle_claim.has_bad_values():
            return None
        rescaling_item_vector, rescaling_iterable_vector = self.compute_rescaling_vectors(enriched_oracle_claim,
                                                                                          ratio_item_iterable_learning)
        self.item_weights_vector = coefficient_wise_vector_product(rescaling_item_vector, self.item_weights_vector)
        self.iterable_weights_vector = coefficient_wise_vector_product(rescaling_iterable_vector,
                                                                       self.iterable_weights_vector)

    def compute_rescaling_vectors(self, enriched_oracle_claim, ratio_item_iterable_learning):
        gradient_item, gradient_iterable = self.compute_item_and_iterable_gradients(enriched_oracle_claim,
                                                                                    ratio_item_iterable_learning)
        gradient_item = rescale_vector_from_gradient_and_effort(gradient_item, enriched_oracle_claim.effort)
        gradient_iterable = rescale_vector_from_gradient_and_effort(gradient_iterable, enriched_oracle_claim.effort)
        return gradient_item, gradient_iterable

    def compute_item_and_iterable_gradients(self, enriched_oracle_claim, ratio_item_iterable_learning):
        eoc = enriched_oracle_claim
        r = ratio_item_iterable_learning
        matrix_of_coefficients = (((1. - eoc.current_distance) * eoc.norm1 / eoc.norm0, -1.),
                                  (-1., (1. - eoc.current_distance) * eoc.norm0 / eoc.norm1))
        vector_of_vectorizations = (eoc.vectorization0, eoc.vectorization1)
        gradient_item = non_trivial_hadamard_scalar_product(vector_of_vectorizations,
                                                            matrix_of_coefficients,
                                                            vector_of_vectorizations)
        u0 = dot_matrix_dot_products(self.iterable_weights_vector, transpose_matrix(self.item_iterable_matrix),
                                     self.item_weights_vector, eoc.vectorization0)
        u1 = dot_matrix_dot_products(self.iterable_weights_vector, transpose_matrix(self.item_iterable_matrix),
                                     self.item_weights_vector, eoc.vectorization1)
        gradient_iterable = non_trivial_hadamard_scalar_product((eoc.iterables_vector0, eoc.iterables_vector1),
                                                                matrix_of_coefficients,
                                                                (u0, u1))
        common_factor = (eoc.norm0 * eoc.norm1 * (eoc.target_distance - eoc.current_distance)
                         / (r ** 2 * norm(gradient_item) ** 2 + (1. - r) ** 2 * norm(gradient_iterable) ** 2))
        gradient_item *= common_factor * r
        gradient_iterable *= common_factor * (1. - r)
        return gradient_item, gradient_iterable


class EnrichedOracleClaim:

    def __init__(self, oracle_claim, distance, effort=1.):
        self.effort = effort
        self.iterables0, self.iterables1 = oracle_claim.iterables_pair
        self.distance_interval = oracle_claim.distance_interval
        self.current_distance, self.iterables_vector0, self.vectorization0, self.norm0,\
            self.iterables_vector1, self.vectorization1, self.norm1 = \
            distance.verbose_distance(self.iterables0, self.iterables1)
        self.target_distance = closest_point_from_interval(self.current_distance, self.distance_interval)
        self.target_distance = (self.current_distance + self.effort * (self.target_distance - self.current_distance))

    def has_bad_values(self):
        return (math.isclose(self.current_distance, self.target_distance)
                or math.isclose(self.norm0, 0) or math.isclose(self.norm1, 0))


def closest_point_from_interval(value, interval):
    lower_bound, upper_bound = interval
    if value < lower_bound:
        return lower_bound
    if value > upper_bound:
        return upper_bound
    return value


def rescale_vector_from_gradient_and_effort(gradient, effort):
    gradient = rescale_vector_to_satisfy_lower_negative_bound(gradient, -1. * effort)
    one_vector = one_vector_from_length(len(gradient))
    return one_vector + gradient


def non_trivial_hadamard_scalar_product(left_vectors, matrix, right_vectors):
    """
    :param left_vectors: a vector of n vectors
    :param matrix: an n by m matrix of scalars
    :param right_vectors: a vector of m vectors
    :return: transpose(left_vector) * matrix * right_vector,
             where the product of vectors is coefficient_wise_vector_product.
    """
    return sum(matrix[i][j] * coefficient_wise_vector_product(left_vectors[i], right_vectors[j])
               for i in range(len(left_vectors)) for j in range(len(right_vectors)))
