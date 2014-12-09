#!/usr/bin/python
# -*- coding: utf-8 -*-

import shell
import util
import wordsegUtil


############################################################
# Problem 1b: Solve the segmentation problem under a unigram model

class SegmentationProblem(util.SearchProblem):

    def __init__(self, query, unigramCost):
        self.query = query
        self.unigramCost = unigramCost

    # State is the current index of the query

    def startState(self):

        # BEGIN_YOUR_CODE (around 5 lines of code expected)

        return 0

        # END_YOUR_CODE

    def isGoal(self, state):

        # BEGIN_YOUR_CODE (around 5 lines of code expected)

        return state == len(self.query)

        # END_YOUR_CODE

    # (action, new state, cost)

    def succAndCost(self, state):

        # BEGIN_YOUR_CODE (around 10 lines of code expected)

        return [(i, i, self.unigramCost(self.query[state:i])) for i in
                range(state, len(self.query) + 1)]


        # END_YOUR_CODE

def segmentWords(query, unigramCost):

    # BEGIN_YOUR_CODE (around 5 lines of code expected)

    if len(query) == 0:
        return ''

    if query == None:
        return None

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(SegmentationProblem(query, unigramCost))

    a = b = 0
    splits = []
    for i in ucs.actions:
        (a, b) = (b, i)
        splits.append(query[a:b])
    return ' '.join(splits)


    # END_YOUR_CODE

############################################################
# Problem 2b: Solve the vowel insertion problem under a bigram cost

class VowelInsertionProblem(util.SearchProblem):

    def __init__(
        self,
        queryWords,
        bigramCost,
        possibleFills,
        ):

        self.queryWords = queryWords
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):

        # BEGIN_YOUR_CODE (around 5 lines of code expected)

        return (0, wordsegUtil.SENTENCE_BEGIN)

        # END_YOUR_CODE

    def isGoal(self, state):

        # BEGIN_YOUR_CODE (around 5 lines of code expected)

        return state[0] == len(self.queryWords)

        # END_YOUR_CODE

    # (action, new state, cost)

    def succAndCost(self, state):

        # BEGIN_YOUR_CODE (around 10 lines of code expected)

        candidates = self.possibleFills(self.queryWords[state[0]])
        if len(candidates)==0:
            candidates.add(self.queryWords[state[0]])
        return [(i, (state[0] + 1, i), self.bigramCost(state[1], i))
                for i in candidates]


        # END_YOUR_CODE

def insertVowels(queryWords, bigramCost, possibleFills):

    # BEGIN_YOUR_CODE (around 5 lines of code expected)

    if queryWords == None:
        return None
    if len(queryWords) == 0:
        return ''

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(VowelInsertionProblem(queryWords, bigramCost,
              possibleFills))
    return ' '.join(ucs.actions)


    # END_YOUR_CODE

############################################################
# Problem 3b: Solve the joint segmentation-and-insertion problem

class JointSegmentationInsertionProblem(util.SearchProblem):

    def __init__(
        self,
        query,
        bigramCost,
        possibleFills,
        ):

        self.query = query
        self.bigramCost = bigramCost
        self.possibleFills = possibleFills

    def startState(self):

        # BEGIN_YOUR_CODE (around 5 lines of code expected)

        return (0, wordsegUtil.SENTENCE_BEGIN)

        # END_YOUR_CODE

    def isGoal(self, state):

        # BEGIN_YOUR_CODE (around 5 lines of code expected)

        return state[0] == len(self.query)

        # END_YOUR_CODE

    # (action, new state, cost)

    def succAndCost(self, state):

        # BEGIN_YOUR_CODE (around 15 lines of code expected)

        retVal = []
        for i in range(state[0], len(self.query) + 1):
            candidates = self.possibleFills(self.query[state[0]:i])
            if len(candidates) == 0:
                continue
            for c in candidates:
                retVal.append((c, (i, c), self.bigramCost(state[1], c)))
        return retVal


        # END_YOUR_CODE

def segmentAndInsert(query, bigramCost, possibleFills):
    if len(query) == 0:
        return ''

    # BEGIN_YOUR_CODE (around 5 lines of code expected)

    if query == None:
        return None

    ucs = util.UniformCostSearch(verbose=0)
    ucs.solve(JointSegmentationInsertionProblem(query, bigramCost,
              possibleFills))
    return ' '.join(ucs.actions)


    # END_YOUR_CODE

############################################################

if __name__ == '__main__':
    shell.main()
