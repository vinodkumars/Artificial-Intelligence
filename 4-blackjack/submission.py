#!/usr/bin/python
# -*- coding: utf-8 -*-

import collections
import util
import math
import random
import sys


############################################################

############################################################
# Problem 2a

class ValueIteration(util.MDPAlgorithm):

    # Implement value iteration.  First, compute V_opt using the methods
    # discussed in class.  Once you have computed V_opt, compute the optimal
    # policy pi.  Note that ValueIteration is an instance of util.MDPAlgrotithm,
    # which means you will need to set pi and V (see util.py).

    def solve(self, mdp, epsilon=0.001):
        mdp.computeStates()
        self.pi = {}
        self.V = {}

        # BEGIN_YOUR_CODE (around 15 lines of code expected)

        maxDiff = epsilon + 1
        V_prev = {}
        while maxDiff >= epsilon:
            for s in mdp.states:
                maxV = sys.maxint * -1
                for a in mdp.actions(s):
                    accumulator = 0
                    for (newState, prob, reward) in \
                        mdp.succAndProbReward(s, a):
                        accumulator += prob * (reward + mdp.discount()
                                * V_prev.get(newState, 0))
                    if accumulator > maxV:
                        maxV = accumulator
                        maxA = a
                self.V[s] = maxV
                self.pi[s] = maxA

            maxDiff = max([abs(self.V.get(s, 0) - V_prev.get(s, 0))
                          for s in mdp.states])
            V_prev = self.V.copy()


        # END_YOUR_CODE
                # (self.V[s], self.pi[s]) = max([(sum([prob * (reward
                #         + mdp.discount() * V_prev.get(newState,
                #         0)) for (newState, prob, reward) in
                #         mdp.succAndProbReward(s, a)]), a) for a in
                #         mdp.actions(s)])

############################################################
# Problem 2b

# If you decide 2b is true, prove it in writeup.pdf and put "return None" for
# the code blocks below.  If you decide that 2b is false, construct a
# counterexample by filling out this class and returning an alpha value in
# counterexampleAlpha().

class CounterexampleMDP(util.MDP):

    def __init__(self, n=3):

        # BEGIN_YOUR_CODE (around 5 lines of code expected)

        self.n = n

        # END_YOUR_CODE

    def startState(self):

        # BEGIN_YOUR_CODE (around 5 lines of code expected)

        return 0

        # END_YOUR_CODE

    # Return set of actions possible from |state|.

    def actions(self, state):

        # BEGIN_YOUR_CODE (around 5 lines of code expected)

        return [+1, -1]

        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.

    def succAndProbReward(self, state, action):

        # BEGIN_YOUR_CODE (around 5 lines of code expected)

        def T(s, a, sDash):
            if (s + a) % self.n == sDash % self.n:
                return 0.9
            else:
                return 0.1

        def TDash(s, a, sDash):
            return T(s, a, sDash) + counterexampleAlpha() / sum([T(s,
                    a, sDash1) + counterexampleAlpha() for sDash1 in
                    range(self.n)])

        return [((state + 1) % self.n, TDash(state, action, (state + 1)
                % self.n), 1), ((state - 1) % self.n, TDash(state,
                action, (state - 1) % self.n), 100)]

        # END_YOUR_CODE

    def discount(self):

        # BEGIN_YOUR_CODE (around 5 lines of code expected)

        return 0.5


        # END_YOUR_CODE

def counterexampleAlpha():

    # BEGIN_YOUR_CODE (around 5 lines of code expected)

    return 0.5


    # END_YOUR_CODE

# mdp = CounterexampleMDP()
# alg = ValueIteration()
# alg.solve(mdp)
# print alg.V

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):

    def __init__(
        self,
        cardValues,
        multiplicity,
        threshold,
        peekCost,
        ):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """

        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.  The second element is the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.  The final element
    # is the current deck.

    def startState(self):
        return (0, None, (self.multiplicity, ) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.

    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to (0,).

    def succAndProbReward(self, state, action):

        # BEGIN_YOUR_CODE (around 55 lines of code expected)

        def removeCardFromDeck(curDeck, index):
            tmpList = list(curDeck)
            tmpList[index] -= 1
            newDeck = tuple(tmpList)

            # If deck is empty, set it to None

            if sum(newDeck) == 0:
                newDeck = None
            return newDeck

        def getNewDeckAndReward(
            curDeck,
            takeIndex,
            newHandVal,
            threshold,
            ):

            reward = 0

            # Check for bust

            if newHandVal > threshold:
                newDeck = None
            else:
                newDeck = removeCardFromDeck(curDeck, takeIndex)

                # If deck is empty, set reward

                if newDeck == None:
                    reward = newHandVal
            return (newDeck, reward)

        # If deck is over

        if state[2] == None:
            return []

        if action == 'Quit':
            return [((state[0], None, None), 1.0, state[0])]

        if action == 'Peek':

            # No double peeking!

            if state[1] != None:
                return []

            retVal = []
            cardSum = sum(state[2])
            for pi in range(len(state[2])):
                if state[2][pi] > 0:
                    retVal.append(((state[0], pi, state[2]),
                                  state[2][pi] * 1.0 / cardSum,
                                  self.peekCost * -1))

            return retVal

        if action == 'Take':

            # If peeked previously, we need to take the peeked card

            if state[1] != None:
                reward = 0
                newHandVal = state[0] + self.cardValues[state[1]]
                (nd, reward) = getNewDeckAndReward(state[2], state[1],
                        newHandVal, self.threshold)
                return [((newHandVal, None, nd), 1.0, reward)]
            else:
                retVal = []
                cardSum = sum(state[2])
                for ti in range(len(state[2])):
                    if state[2][ti] > 0:
                        newHandVal = state[0] + self.cardValues[ti]
                        (nd, reward) = getNewDeckAndReward(state[2],
                                ti, newHandVal, self.threshold)
                        retVal.append(((newHandVal, None, nd),
                                state[2][ti] * 1.0 / cardSum, reward))
                return retVal

        # END_YOUR_CODE

    def discount(self):
        return 1.0


############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """

    # BEGIN_YOUR_CODE (around 5 lines of code expected)

    return BlackjackMDP(cardValues=[2, 3, 21], multiplicity=2,
                        threshold=20, peekCost=1)


    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action

class QLearningAlgorithm(util.RLAlgorithm):

    def __init__(
        self,
        actions,
        discount,
        featureExtractor,
        explorationProb=0.2,
        ):

        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = collections.Counter()
        self.numIters = 0

    # Return the Q function associated with the weights and features

    def getQ(self, state, action):
        score = 0
        for (f, v) in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.

    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in
                       self.actions(state))[1]

    # Call this function to get the step size to update the weights.

    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.

    def incorporateFeedback(
        self,
        state,
        action,
        reward,
        newState,
        ):

        # BEGIN_YOUR_CODE (around 15 lines of code expected)

        if newState == None:
            return
        residual = reward + self.discount * max([self.getQ(newState,
                aDash) for aDash in self.actions(newState)]) \
            - self.getQ(state, action)
        for (f, v) in self.featureExtractor(state, action):
            self.weights[f] = self.weights[f] + self.getStepSize() \
                * residual * v


        # END_YOUR_CODE

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.

def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]


############################################################
# Problem 4b: convergence of Q-learning

# Small test case

smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                        threshold=10, peekCost=1)

# Large test case

largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3,
                        threshold=40, peekCost=1)


# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action (1 feature).  Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).  Only add these features if the deck != None

def blackjackFeatureExtractor(state, action):
    if state == None:
        return []
    (total, nextCard, counts) = state

    # BEGIN_YOUR_CODE (around 10 lines of code expected)

    phi = []
    phi.append(((total, action), 1))
    if counts != None:
        cardPresence = tuple([int(x > 0) for x in counts])
        phi.append(((cardPresence, action), 1))
        for i in range(len(counts)):
            phi.append(((i, counts[i], action), 1))
    return phi


    # END_YOUR_CODE

############################################################
# Problem 4d: changing mdp

# Original mdp

originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                           threshold=10, peekCost=1)

# New threshold

newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2,
                               threshold=15, peekCost=1)

############################################################
# Problem 4b,4c:

# mdp = largeMDP
# mdp.computeStates()
# alg = QLearningAlgorithm(mdp.actions, mdp.discount(),
#                          identityFeatureExtractor, 0.2)
# util.simulate(mdp, alg, numTrials=30000, maxIterations=1000)
# alg.explorationProb = 0
# QLearnPolicy = {}
# for s in mdp.states:
#     QLearnPolicy[s] = alg.getAction(s)

## print 'QLearn:'
## print QLearnPolicy

# alg = ValueIteration()
# alg.solve(mdp, 0.0001)
# valueIteractionPolicy = alg.pi

## print 'Value iteration'
## print valueIteractionPolicy

# if valueIteractionPolicy == QLearnPolicy:
#     print 'Match'
# else:
#     notMatchCount = 0
#     for s in mdp.states:
#         if valueIteractionPolicy.get(s) != QLearnPolicy.get(s):
#             notMatchCount += 1

#             # print 'Not a match at state: {}'.format(s)
#             # print 'QLearnPolicy: {} \t VIPolicy: {}'.format(QLearnPolicy.get(s), valueIteractionPolicy.get(s))

#     print 'Not a match for {} states out of {}'.format(notMatchCount,
#             len(QLearnPolicy))

############################################################
# Problem 4d:

# originalMDP.computeStates()
# newThresholdMDP.computeStates()
# alg = ValueIteration()
# alg.solve(originalMDP, 0.0001)
# fixedRL = util.FixedRLAlgorithm(alg.pi)
# VIReward = util.simulate(newThresholdMDP, fixedRL, numTrials=30000)
# alg = QLearningAlgorithm(newThresholdMDP.actions,
#                          newThresholdMDP.discount(),
#                          blackjackFeatureExtractor, 0.2)
# QLReward = util.simulate(newThresholdMDP, alg, numTrials=30000)
# print 'VI reward: {}'.format(sum(VIReward)*1.0/len(VIReward))
# print 'QL reward: {}'.format(sum(QLReward)*1.0/len(QLReward))

############################################################

# 3b
# mdp = peekingMDP()
# vi = ValueIteration()
# vi.solve(mdp)
# f = len([a for a in vi.pi.values() if a == 'Peek']) / float(len(vi.pi.values()))
# print f

# 4c
# mdp = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10,
#                    peekCost=1)
# mdp.computeStates()
# rl = QLearningAlgorithm(mdp.actions, mdp.discount(),
#                         blackjackFeatureExtractor, 0)
# rl.numIters = 1
# print rl.getQ((7, None, (0, 1)), 'Quit')
