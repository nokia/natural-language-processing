# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
from learning_distance import *
from oracle_claim import OracleClaim

iterables = ['aa', 'ab', 'bbb']
item_to_weight = {'a': 1, 'b': 2}
iterable_to_weight = {'aa': 1, 'ab': 2, 'bbb': 3}
distance = LearningDistance(iterables, item_to_weight, iterable_to_weight)
iterables0 = {'ab'}
iterables1 = {'bbb'}
iterables2 = {'ab', 'ab'}
iterables3 = {'bbb', 'aa'}


class TestLearningDistance(unittest.TestCase):

    def test_learn_correct_interval_target(self):
        iterable0 = ('a', 'a', 'a', 'b')
        iterable1 = ('a', 'a', 'a', 'b')
        iterable2 = ('a', 'a', 'b', 'b')
        iterable3 = ('a', 'c')
        all_iterables = [iterable0, iterable1, iterable2, iterable3]
        item_weights = {'a': 1., 'b': 2., 'c': 0.5}
        iterable_weights = {iterable0: 1., iterable1: 1., iterable2: 1., iterable3: 1.}
        interval_true_distance = (0.1, 0.2)
        oracle_claim = OracleClaim(({iterable0, iterable1}, {iterable1, iterable3}), interval_true_distance)
        learning_distance = LearningDistance(all_iterables, item_to_weight=item_weights,
                                             iterable_to_weight=iterable_weights)
        # old_distance = learning_distance(oracle_claim.iterables_pair[0], oracle_claim.iterables_pair[1])
        learning_distance.learn({oracle_claim}, ratio_item_iterable_learning=1., number_of_iterations=5,
                                convergence_speed=0.5)
        new_distance = learning_distance(oracle_claim.iterables_pair[0], oracle_claim.iterables_pair[1])
        self.assertTrue(abs(new_distance - 0.1) < abs(new_distance - 0.2))

    def test_learn(self):
        current_distance0 = distance(iterables0, iterables1)
        target_distance0 = current_distance0 * 2.
        oracle_claim0 = OracleClaim((iterables0, iterables1), (target_distance0, 1.))
        current_distance1 = distance(iterables2, iterables3)
        target_distance1 = current_distance1 * 2.
        oracle_claim1 = OracleClaim((iterables2, iterables3), (target_distance1, 1.))
        distance.learn([oracle_claim0, oracle_claim1], number_of_iterations=5)
        obtained_distance0 = distance(iterables0, iterables1)
        obtained_distance1 = distance(iterables2, iterables3)
        self.assertTrue(abs(obtained_distance0 - target_distance0) < abs(current_distance0 - target_distance0))
        self.assertTrue(abs(obtained_distance1 - target_distance1) < abs(current_distance1 - target_distance1))

    def test_learn_from_one_oracle_claim_larger_distance(self):
        current_distance = distance(iterables0, iterables1)
        target_distance = current_distance * 2.
        oracle_claim = OracleClaim((iterables0, iterables1), (target_distance, 1.))
        distance.learn_from_one_oracle_claim(oracle_claim, effort=0.5)
        obtained_distance = distance(iterables0, iterables1)
        self.assertTrue(abs(obtained_distance - target_distance) < abs(current_distance - target_distance))

    def test_learn_from_one_oracle_claim_smaller_distance(self):
        current_distance = distance(iterables0, iterables1)
        target_distance = current_distance / 2.
        oracle_claim = OracleClaim((iterables0, iterables1), (0., target_distance))
        distance.learn_from_one_oracle_claim(oracle_claim, effort=0.5)
        obtained_distance = distance(iterables0, iterables1)
        self.assertTrue(abs(obtained_distance - target_distance) < abs(current_distance - target_distance))

    def test_closest_point_from_interval(self):
        interval = (-2, 4)
        self.assertEqual(closest_point_from_interval(-3, interval), -2)
        self.assertEqual(closest_point_from_interval(-2, interval), -2)
        self.assertEqual(closest_point_from_interval(5, interval), 4)
        self.assertEqual(closest_point_from_interval(1, interval), 1)

    def test_non_trivial_hadamard_scalar_product(self):
        matrix = ((1, 2), (3, 4))
        u0 = create_vector([1, 2, 3])
        u1 = create_vector([10, 20, 30])
        v0 = create_vector([4, 1, 2])
        v1 = create_vector([3, 2, 1])
        computed = non_trivial_hadamard_scalar_product((u0, u1), matrix, (v0, v1))
        r00 = coefficient_wise_vector_product(u0, v0)
        r01 = coefficient_wise_vector_product(u0, v1)
        r10 = coefficient_wise_vector_product(u1, v0)
        r11 = coefficient_wise_vector_product(u1, v1)
        expected = matrix[0][0] * r00 + matrix[0][1] * r01 + matrix[1][0] * r10 + matrix[1][1] * r11
        self.assertTrue(are_equal_vectors(expected, computed))

    def test_rescale_vector_from_gradient_and_effort(self):
        vector = create_vector([6, 4, -2, 0])
        computed = rescale_vector_from_gradient_and_effort(vector, 1.)
        expected = create_vector([1 + 6/2, 1 + 4/2, 1 + -2/2, 1])
        self.assertTrue(are_equal_vectors(expected, computed))
        vector = create_vector([0, 1, 2, 3])
        computed = rescale_vector_from_gradient_and_effort(vector, 1.)
        expected = create_vector([1, 2, 3, 4])
        self.assertTrue(are_equal_vectors(expected, computed))


def random_vector(length):
    return create_vector([0.5 - random.random() for _ in range(length)])


if __name__ == '__main__':
    unittest.main()


"""
# -------------   DRAFT   --------------- #

from learning_distance import LearningDistance
from oracle_claim import OracleClaim

def normalize_distribution(distribution):
    normalization_factor = sum(distribution.values())
    return {item: value / normalization_factor for item, value in distribution.items()}
    
    
iterable0 = ('a', 'a', 'a', 'b')
iterable1 = ('a', 'a', 'a', 'b')
iterable2 = ('a', 'a', 'b', 'b')
iterable3 = ('a', 'c')
all_iterables = [iterable0, iterable1, iterable2, iterable3]
item_weights = {'a': 1., 'b': 2., 'c': 0.5}
iterable_weights = {iterable0: 1., iterable1: 1., iterable2: 1., iterable3: 1.}
interval_true_distance = (0.1, 0.2)
oracle_claim = OracleClaim(({iterable0, iterable1}, {iterable1, iterable3}), interval_true_distance)
learning_distance = LearningDistance(all_iterables, item_to_weight=item_weights,
                                        iterable_to_weight=iterable_weights)
print(normalize_distribution(learning_distance.get_item_weights()))
old_distance = learning_distance(oracle_claim.iterables_pair[0], oracle_claim.iterables_pair[1])
print(old_distance)
learning_distance.learn({oracle_claim}, ratio_item_iterable_learning=1., number_of_iterations=5,
                        convergence_speed=0.5)
print(normalize_distribution(learning_distance.get_item_weights()))
new_distance = learning_distance(oracle_claim.iterables_pair[0], oracle_claim.iterables_pair[1])
print(new_distance)



iterable0 = ('a', 'a', 'a', 'b')
iterable1 = ('a', 'a', 'a', 'b')
iterable2 = ('a', 'a', 'b', 'b')
iterable3 = ('a', 'c')
all_iterables = [iterable0, iterable1, iterable2, iterable3]
item_weights = {'a': 1., 'b': 2., 'c': 0.5}
iterable_weights = {iterable0: 1., iterable1: 1., iterable2: 1., iterable3: 1.}
learning_distance = LearningDistance(all_iterables, item_to_weight=item_weights, iterable_to_weight=iterable_weights)
print(normalize_distribution(learning_distance.get_item_weights()))
# {'a': 0.2857142857142857, 'b': 0.5714285714285714, 'c': 0.14285714285714285}
print(learning_distance({iterable0, iterable1}, {iterable1, iterable3}))
# 0.004282674925764063

oracle_claim = OracleClaim(({iterable0, iterable1}, {iterable1, iterable3}), (0.1, 0.2))
set_of_claims = {oracle_claim}
learning_distance.learn(set_of_claims, ratio_item_iterable_learning=1., number_of_iterations=100, convergence_speed=0.1)
print(normalize_distribution(learning_distance.get_item_weights()))

print(learning_distance({iterable0, iterable1}, {iterable1, iterable3}))
# 0.2007295884897502 ???


vector delta (unitary?)
distance to target
distance to setting a weight to 0
number of steps
speed of convergence
"""