# natural-language-processing
# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


============ Summary ============

This Python package provides a distance on collections of iterables
that is able to evolve when true distance instances are provided.
Its main application is expected to be on Natural Language Processing.
It unifies the 'bag of words' and 'bag of factors' approaches.
Indeed, a text can be transformed into a collection of words after lemmatization,
or into a the collection of its factors (bounding the length of those factors
is then recommended).

The LearningDistance object is initialized by providing
the collection of all iterables considered.
The distance itself is computed using weights on the items and iterables
(a large weight corresponding to an important item or iterable).
Those weights can be provided by the user.
Otherwise, they are computed following the tf-idf heuristic.

The LearningDistance object is also able to learn.
Let us define an 'oracle claim' as a pair of collections of iterables
and an interval for the true distance between them.
The LearningDistance object can use a collection of oracle claims
to change the item and iterable weights so that
the distance conforms better with the oracle claims.

The distance between two collections of iterables is defined as
the cosine distance between their vectorizations.
The vectorization of a collection of iterables is a vector on the space
of all items, which coefficients depends on the item and iterable weights,
following the classic 'bag of words' and 'tf-idf' approaches.

All the computations rely on sparse matrix manipulations, implemented by scipy.


============ Tutorial ============

# Import the main class.

from learning_distance import LearningDistance

# Define the iterables we are working on. They must be hashable.

iterable0 = ('a', 'b', 'a', 'a')
iterable1 = ('a', 'a', 'a', 'b')
iterable2 = ('a', 'b', 'b', 'a')
iterable3 = ('a', 'c')
all_iterables = [iterable0, iterable1, iterable2, iterable3]

# The order of the items does not impact the distance computation,
# but their multiplicity does.

# Define weights on the items and iterables.

item_weights = {'a': 1., 'b': 2., 'c': 0.5}
iterable_weights = {iterable0: 4.5, iterable1: 1., iterable2: 3., iterable3: 2.5}

# Define the LearningDistance object.

learning_distance = LearningDistance(all_iterables, item_to_weight=item_weights, iterable_to_weight=iterable_weights)

# If the 'item_weights' and / or 'iterable_weights' are omitted,
# default weights will be computed using the tf-idf heuristic.

# The LearningDistance object is callable and returns
# the distance between its arguments.
# Recall that those arguments are collections of iterables,
# and not iterables alone.

learning_distance({iterable0, iterable1}, {iterable1, iterable3})
# 0.04990974137709514

# Maybe we find this distance too low and expected a value
# between 0.1 and 0.2 instead. To express it, we make the following oracle claim.

from oracle_claim import OracleClaim
oracle_claim = OracleClaim(({iterable0, iterable1}, {iterable1, iterable3}), (0.1, 0.2))

# We can now let learning_distance learn from this claim
# and adjust its weights on items and iterables.
# Let us say that we want the changes to be mainly on the item weights,
# then we write

set_of_claims = {oracle_claim}
learning_distance.learn(set_of_claims, ratio_item_iterable_learning=1.)

# The collection of claims can be a set or a list.
# We can now check that the distance has changed.

learning_distance({iterable0, iterable1}, {iterable1, iterable3})
# 0.10104605065402616

# The item and iterable weights can be accessed either as dictionary

learning_distance.get_item_weights()
# {'a': 0.7187108156315916, 'b': 2.22528447611958, 'c': 0.5843234731543091}

learning_distance.get_iterable_weights()
"""{('a', 'b', 'a', 'a'): 4.5,
    ('a', 'a', 'a', 'b'): 0.9403565677689033,
    ('a', 'b', 'b', 'a'): 3.0,
    ('a', 'c'): 2.6491085805777415}"""

or as vectors, using directly the attributes

learning_distance.item_weights_vector
# array([0.71871082, 2.22528448, 0.58432347])

learning_distance.iterable_weights_vector
# array([4.5, 0.94035657, 3., 2.64910858])

# This second option is faster, but we do not know to which item (or iterable)
# correspond the indices of the vectors.
# Those indices can be obtained using the dictionaries

learning_distance.item_to_index
# {'a': 0, 'b': 1, 'c': 2}

learning_distance.iterable_to_index
"""{('a', 'b', 'a', 'a'): 0,
    ('a', 'a', 'a', 'b'): 1,
    ('a', 'b', 'b', 'a'): 2,
    ('a', 'c'): 3}"""


============ Structure ============


--- matrix_operations.py ---

Define the functions manipulating vectors and matrices.
Rely on scipy sparse matrices.
Only this file needs changing if another implementation is chosen in the future.


--- vector_space.py ---

Define the class 'VectorSpace', initialized using a collection of iterables,
and transforming item or iterable collections into vectors.
Provide the methods
    __init__(iterables)
    item_vector_from_dict(self, item_distribution)
    iterable_vector_from_dict(self, iterable_distribution)
    item_dict_from_vector(self, item_vector)
    iterable_dict_from_vector(self, iterable_vector)
    iterable_vector_from_collection(self, iterable_collection)
    count_iterables_containing_item(self, item)


--- distance.py ---

Define the class 'Distance'. Objects of this class are callable.
They input pairs of collections of iterables and output their distance.
Provide the methods
    __init__(self, iterables, item_to_weight=None, iterable_to_weight=None)
    def __call__(self, iterables0, iterables1)
    vectorize(self, iterables)
    set_item_weights(self, item_to_weight)
    set_iterable_weights(self, iterable_to_weight)
    get_item_weights(self)
    get_iterable_weights(self)
    tfidf_item_weights(self)
    verbose_distance(self, iterables0, iterables1)
    verbose_vectorize(self, iterables)


--- oracle_claim.py ---

Define the class 'OracleClaim', which is used to provide
bounds on the distance between two collections of iterables.
Provide the method
    __init__(self, iterables_pair, distance_interval)


--- learning_distance.py ---

Define the class 'LearningDistance', which inherits from 'Distance'.
Add the functionality to learn from 'OracleClaim' objects.
Provide the methods
    __init__(self, iterables, item_to_weight=None, iterable_to_weight=None)
    learn(self, oracle_claims, ratio_item_iterable_learning=0.5, convergence_speed=0.5,
          number_of_iterations=DEFAULT_NUMBER_OF_ITERATIONS)
    learning_loop_on_oracle_claims(self, oracle_claims, ratio_item_iterable_learning=0.5, effort=1.)
    learn_from_one_oracle_claim(self, oracle_claim, ratio_item_iterable_learning=0.5, effort=1.)
    compute_rescaling_vectors(self, enriched_oracle_claim, ratio_item_iterable_learning)

Also define the class 'EnrichedOracleClaim', used to avoid
duplicate computations during the treatment of an oracle claim.


--- tests ---

Contain the unittests for the various files.
