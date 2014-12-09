#!/usr/bin/python
# -*- coding: utf-8 -*-

import random
import collections
import math
import sys
from collections import Counter
from util import *


############################################################
# Problem 2: binary classification
############################################################

############################################################
# Problem 2a: feature extraction

def extractWordFeatures(x):
    """
    Extract word features for a string x.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """

    # BEGIN_YOUR_CODE (around 5 lines of code expected)

    return dict(collections.Counter(x.split()))


    # END_YOUR_CODE

############################################################
# Problem 2b: stochastic gradient descent

def learnPredictor(trainExamples, testExamples, featureExtractor):
    '''
    Given |trainExamples| and |testExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of iterations to
    train |numIters|, return the weight vector (sparse feature vector) learned.

    You should implement stochastic gradient descent.

    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and testExamples
    to see how you're doing as you learn after each iteration.
    '''

    weights = {}  # feature => weight
    phiCollection = {}

    # hyper-parameters for extractWordFeatures

    numIters = 19  # number of iterations
    eta = 0.02  # step size

    # hyper-parameters for extractCharacterFeatures(6)
    # numIters = 15  # number of iterations
    # eta = 0.02  # step size

    # hyper-parameters for extractExtraCreditFeatures
    # numIters = 17  # number of iterations
    # eta = 0.02  # step size

    # BEGIN_YOUR_CODE (around 15 lines of code expected)

    def predictor(x):
        if dotProduct(weights, featureExtractor(x)) >= 0:
            return 1
        else:
            return -1

    for i in range(len(trainExamples)):
        phiCollection[i] = featureExtractor(trainExamples[i][0])

    for i in range(numIters):
        for j in range(len(trainExamples)):
            phi = phiCollection[j]
            y = trainExamples[j][1]
            margin = dotProduct(weights, phi) * y
            if margin < 1:
                increment(weights, eta * y, phi)

        print 'train error: ', evaluatePredictor(trainExamples,
                predictor)
        print 'dev error: ', evaluatePredictor(testExamples, predictor)

    # END_YOUR_CODE

    return weights


############################################################
# Problem 2c: generate test case

def generateDataset(numExamples, weights):
    '''
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    '''

    random.seed(42)

    # Return a single example (phi(x), y).
    # phi(x) can be anything (randomize!) with a nonzero score under the given weight vector
    # y should be 1 or -1 as classified by the weight vector.

    def generateExample():

        # BEGIN_YOUR_CODE (around 5 lines of code expected)

        phi = {}
        for k in iter(weights):
            phi[k] = random.randint(0, 10)
        y = (1 if dotProduct(weights, phi) >= 0 else -1)

        # END_YOUR_CODE

        return (phi, y)

    return [generateExample() for _ in range(numExamples)]


############################################################
# Problem 2f: character features

def extractCharacterFeatures(n):
    '''
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    '''

    def extract(x):

        # BEGIN_YOUR_CODE (around 10 lines of code expected)

        xNoSpace = x.replace(' ', '')
        if len(xNoSpace) <= n:
            return dict(collections.Counter([xNoSpace]))
        return dict(collections.Counter([xNoSpace[i:i + n] for i in
                    range(len(xNoSpace) - n + 1)]))

        # END_YOUR_CODE

    return extract


############################################################
# Problem 2h: extra credit features

wordNGramsN = 3


def extractExtraCreditFeatures(x):

    # BEGIN_YOUR_CODE (around 5 lines of code expected)

    splits = x.split()
    d = extractWordFeatures(x)
    d.update(collections.Counter([' '.join(splits[i:i + wordNGramsN])
             for i in range(len(splits) - wordNGramsN + 1)]))

    return d


    # END_YOUR_CODE

############################################################
# Problem 3: k-means
############################################################

def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters
    maxIters: maximum number of iterations to run for (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments, (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''

    # BEGIN_YOUR_CODE (around 35 lines of code expected)

    centers = {}
    assignments = {}
    centersSumOfSquares = {}
    random.seed(42)
    randset = set()
    for i in range(K):
        while True:
            r = random.randint(0, len(examples))
            if r not in randset:
                centers[i] = examples[r]
                randset.add(r)
                break

    def getSquaredDist(pt1, pt2):
        dist = 0
        for x in pt1:
            dist = dist + (pt1.get(x, 0) - pt2.get(x, 0)) ** 2
        for x in pt2:
            if x not in pt1:
                dist = dist + pt2.get(x, 0) ** 2
        return dist

    def getSumOfSquares(pt):
        return sum([pt.get(x) ** 2 for x in pt])

    for i in range(maxIters):

        prevAssignments = assignments.copy()

        for j in range(len(centers)):
            centersSumOfSquares[j] = getSumOfSquares(centers[j])

        for j in range(len(examples)):
            minval = sys.maxint
            for k in range(len(centers)):
                dist = centersSumOfSquares[k] - 2 \
                    * dotProduct(examples[j], centers[k])
                if dist < minval:
                    assignments[j] = k
                    minval = dist

        if prevAssignments == assignments:
            break

        for j in range(len(centers)):
            assignedExamples = [x for x in assignments
                                if assignments.get(x, 0) == j]
            centers[j] = dict()
            for k in assignedExamples:
                increment(centers[j], 1, examples[k])
            for x in centers[j]:
                centers[j][x] = 1.0 * centers[j][x] \
                    / len(assignedExamples)

    reconsLoss = 0
    for i in assignments:
        reconsLoss += getSquaredDist(examples[i],
                centers[assignments[i]])

    return (centers, assignments, reconsLoss)


    # END_YOUR_CODE
