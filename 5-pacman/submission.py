#!/usr/bin/python
# -*- coding: utf-8 -*-

from util import manhattanDistance
from game import Directions
import random
import util
import heapq
import sys

from game import Agent
from ghostAgents import RandomGhost


class ReflexAgent(Agent):

    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """

    def __init__(self):
        self.lastPositions = []
        self.dc = None

    def getAction(self, gameState):
        """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """

    # Collect legal moves and successor states

        legalMoves = gameState.getLegalActions()

    # Choose one of the best actions

        scores = [self.evaluationFunction(gameState, action)
                  for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores))
                       if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """

    # Useful information you can extract from a GameState (pacman.py)

        successorGameState = \
            currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in
                          newGhostStates]

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """

    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):

    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):

    """
    Your minimax agent (problem 1)
  """

    def getAction(self, gameState):
        """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following: 
      pacman won, pacman lost or there are no legal moves. 

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
....
      gameState.isWin():
        Returns True if it's a winning state
....
      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue
    """

    # BEGIN_YOUR_CODE (around 30 lines of code expected)

        def recurse(state, depth, agentIndex):
            if state.isWin() or state.isLose():
                return state.getScore()
            elif depth == 0:
                return self.evaluationFunction(state)
            elif agentIndex == 0:
                choices = [recurse(state.generateSuccessor(agentIndex,
                           action), depth, agentIndex + 1)
                           for action in
                           state.getLegalActions(agentIndex) if action
                           != Directions.STOP]
                if len(choices) > 0:
                    return max(choices)
                else:
                    return state.getScore()
            elif agentIndex == state.getNumAgents() - 1:
                choices = [recurse(state.generateSuccessor(agentIndex,
                           action), depth - 1, 0) for action in
                           state.getLegalActions(agentIndex)]
                if len(choices) > 0:
                    return min(choices)
                else:
                    return state.getScore()
            else:
                choices = [recurse(state.generateSuccessor(agentIndex,
                           action), depth, agentIndex + 1)
                           for action in
                           state.getLegalActions(agentIndex)]
                if len(choices) > 0:
                    return min(choices)
                else:
                    return state.getScore()

        choices = [(recurse(gameState.generateSuccessor(self.index,
                   action), self.depth, self.index + 1), action)
                   for action in gameState.getLegalActions(self.index)
                   if action != Directions.STOP]

        (minimaxValue, bestAction) = max(choices)
        ties = [i for i in choices if i[0] == minimaxValue]
        (minimaxValue, bestAction) = random.choice(ties)

        # print 'minimaxValue={}, bestAction={}'.format(minimaxValue,
        #         bestAction)

        return bestAction


    # END_YOUR_CODE

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):

    """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

    def getAction(self, gameState):
        """
      Returns the minimax action using self.depth and self.evaluationFunction
    """

    # BEGIN_YOUR_CODE (around 50 lines of code expected)

        # return (minimax value, action)

        def recurse(
            state,
            depth,
            alpha,
            beta,
            agentIndex,
            ):

            if state.isWin() or state.isLose():
                return 1.0 * state.getScore()
            elif depth == 0:
                return self.evaluationFunction(state)
            elif agentIndex == 0:
                if len(state.getLegalActions(agentIndex)) <= 0:
                    return 1.0 * state.getScore()
                for action in state.getLegalActions(agentIndex):
                    if action == Directions.STOP:
                        continue
                    alpha = max(alpha,
                                recurse(state.generateSuccessor(agentIndex,
                                action), depth, alpha, beta, agentIndex
                                + 1))
                    if beta <= alpha:
                        break
                return alpha
            elif agentIndex == state.getNumAgents() - 1:
                if len(state.getLegalActions(agentIndex)) <= 0:
                    return 1.0 * state.getScore()
                for action in state.getLegalActions(agentIndex):
                    beta = min(beta,
                               recurse(state.generateSuccessor(agentIndex,
                               action), depth - 1, alpha, beta, 0))
                    if beta <= alpha:
                        break
                return beta
            else:
                if len(state.getLegalActions(agentIndex)) <= 0:
                    return 1.0 * state.getScore()
                for action in state.getLegalActions(agentIndex):
                    beta = min(beta,
                               recurse(state.generateSuccessor(agentIndex,
                               action), depth, alpha, beta, agentIndex
                               + 1))
                    if beta <= alpha:
                        break
                return beta

        choices = [(recurse(gameState.generateSuccessor(self.index,
                   action), self.depth, float('-inf'), float('inf'),
                   self.index + 1), action) for action in
                   gameState.getLegalActions(self.index) if action
                   != Directions.STOP]

        (minimaxValue, bestAction) = max(choices)
        ties = [i for i in choices if i[0] == minimaxValue]
        (minimaxValue, bestAction) = random.choice(ties)

        # print 'minimaxValue={}, bestAction={}'.format(minimaxValue,
        #         bestAction)

        return bestAction


    # END_YOUR_CODE

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):

    """
    Your expectimax agent (problem 3)
  """

    def getAction(self, gameState):
        """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """

    # BEGIN_YOUR_CODE (around 25 lines of code expected)

        def recurse(state, depth, agentIndex):
            if state.isWin() or state.isLose():
                return state.getScore()
            elif depth == 0:
                return self.evaluationFunction(state)
            elif agentIndex == 0:
                choices = [recurse(state.generateSuccessor(agentIndex,
                           action), depth, agentIndex + 1)
                           for action in
                           state.getLegalActions(agentIndex) if action
                           != Directions.STOP]
                if len(choices) > 0:
                    return max(choices)
                else:
                    return state.getScore()
            elif agentIndex == state.getNumAgents() - 1:
                randomGhost = RandomGhost(agentIndex)
                return sum([recurse(state.generateSuccessor(agentIndex,
                           action), depth - 1, 0) * prob for (action,
                           prob) in
                           randomGhost.getDistribution(state).items()])
            else:
                randomGhost = RandomGhost(agentIndex)
                return sum([recurse(state.generateSuccessor(agentIndex,
                           action), depth, agentIndex + 1) * prob
                           for (action, prob) in
                           randomGhost.getDistribution(state).items()])

        choices = [(recurse(gameState.generateSuccessor(self.index,
                   action), self.depth, self.index + 1), action)
                   for action in gameState.getLegalActions(self.index)
                   if action != Directions.STOP]

        # for i in choices:
        #     print i[0], i[1]

        (minimaxValue, bestAction) = max(choices)
        ties = [i for i in choices if i[0] == minimaxValue]
        (minimaxValue, bestAction) = random.choice(ties)

        # print 'minimaxValue={}, bestAction={}'.format(minimaxValue,
        #         bestAction)

        return bestAction


    # END_YOUR_CODE

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (problem 4).

    DESCRIPTION:
    distance to nearest bumping into ghost
    distance to nearest food
    score of state
  """

  # BEGIN_YOUR_CODE (around 30 lines of code expected)

    ucs = UniformCostSearch()
    capsuleList = currentGameState.getCapsules()

    def myFoodUCSsuccAndCost(position):
        retVal = []
        for p in [(position[0] - 1, position[1]), (position[0] + 1,
                  position[1]), (position[0], position[1] - 1),
                  (position[0], position[1] + 1)]:
            if p[0] < 0 or p[0] >= currentGameState.getFood().width:
                continue
            if p[1] < 0 or p[1] >= currentGameState.getFood().height:
                continue
            (match, ghostState) = max([(int(p == gs.getPosition()), gs)
                    for gs in currentGameState.getGhostStates()])
            if match == 1:
                if ghostState.scaredTimer > 0:
                    retVal.append((None, p, 0))
                continue
            if currentGameState.hasWall(p[0], p[1]):
                continue
            if p in capsuleList or currentGameState.hasFood(p[0], p[1]):
                retVal.append((None, p, 0))
            else:
                retVal.append((None, p, 1))
        return retVal

    def myGhostUCSsuccAndCost(position):
        retVal = []
        for p in [(position[0] - 1, position[1]), (position[0] + 1,
                  position[1]), (position[0], position[1] - 1),
                  (position[0], position[1] + 1)]:
            if p[0] < 0 or p[0] >= currentGameState.getFood().width:
                continue
            if p[1] < 0 or p[1] >= currentGameState.getFood().height:
                continue
            if currentGameState.hasWall(p[0], p[1]):
                continue
            retVal.append((None, p, 1))
        return retVal

    def findClosestFoodDist():
        minCost = sys.maxint
        noneCost = 100
        maxRadius = max(currentGameState.getFood().width,
                        currentGameState.getFood().height)
        for r in range(1, maxRadius):
            if minCost <= r:
                return (minCost if minCost != None else noneCost)

            for x in range(currentGameState.getPacmanPosition()[0] - r,
                           currentGameState.getPacmanPosition()[0] + r
                           + 1):
                if x < 0 or x >= currentGameState.getFood().width:
                    continue
                for y in [currentGameState.getPacmanPosition()[1] - r,
                          currentGameState.getPacmanPosition()[1] + r]:
                    if y < 0 or y >= currentGameState.getFood().height:
                        continue
                    if minCost \
                        <= manhattanDistance(currentGameState.getPacmanPosition(),
                            [x, y]):
                        continue
                    if currentGameState.getFood()[x][y] or (x, y) \
                        in capsuleList:
                        ucs.solve(currentGameState.getPacmanPosition(),
                                  (x, y), myFoodUCSsuccAndCost)
                        minCost = min(minCost, ucs.totalCost)

            for x in [currentGameState.getPacmanPosition()[0] - r,
                      currentGameState.getPacmanPosition()[0] + r]:
                if x < 0 or x >= currentGameState.getFood().width:
                    continue
                for y in range(currentGameState.getPacmanPosition()[1]
                               - r,
                               currentGameState.getPacmanPosition()[1]
                               + r + 1):
                    if y < 0 or y >= currentGameState.getFood().height:
                        continue
                    if currentGameState.getFood()[x][y] or (x, y) \
                        in capsuleList:
                        ucs.solve(currentGameState.getPacmanPosition(),
                                  (x, y), myFoodUCSsuccAndCost)
                        minCost = min(minCost, ucs.totalCost)
        return (minCost if minCost != None and minCost
                != sys.maxint else noneCost)

    def findClosestGhostDist():
        minCost = sys.maxint
        noneCost = 100
        for gs in currentGameState.getGhostStates():
            if gs.scaredTimer <= 0:
                ucs.solve(currentGameState.getPacmanPosition(),
                          gs.getPosition(), myGhostUCSsuccAndCost)
                minCost = min(minCost, ucs.totalCost)
        return (minCost if minCost != None and minCost
                != sys.maxint else noneCost)

    def findClosestGhostFoodDist():
        minCost = sys.maxint
        noneCost = 100
        for gs in currentGameState.getGhostStates():
            if gs.scaredTimer > 0:
                ucs.solve(currentGameState.getPacmanPosition(),
                          gs.getPosition(), myFoodUCSsuccAndCost)
                minCost = min(minCost, ucs.totalCost)
        return (minCost if minCost != None and minCost
                != sys.maxint else noneCost)

    # def findClosestCapsulesDist():
    #     minCost = sys.maxint
    #     noneCost = 25
    #     for i in currentGameState.getCapsules():
    #         ucs.solve(currentGameState.getPacmanPosition(), i,
    #                   myFoodUCSsuccAndCost)
    #         minCost = min(minCost, ucs.totalCost)
    #     return (minCost if minCost != None and minCost
    #             != sys.maxint else noneCost)

    # dist to food, dist to ghost, score

    # print findClosestFoodDist()
    # print findClosestCapsulesDist()

    phi = [1.0 / (findClosestFoodDist() + 1), 1.0
           / (findClosestGhostFoodDist() + 1), findClosestGhostDist(),
           currentGameState.getScore()]
    weight = [15, 20, 1, 1]

    # weight = [10, 2, 1] 50
    # weight = [10, 1, 1] 100

    retVal = 0
    for i in range(len(phi)):
        retVal += 1.0 * phi[i] * weight[i]

    # print retVal

    return retVal


  # END_YOUR_CODE

# Abbreviation

better = betterEvaluationFunction


######################################################################################
# Uniform cost search algorithm (Dijkstra's algorithm).

class UniformCostSearch:

    def __init__(self, verbose=0):
        self.verbose = verbose

    def solve(
        self,
        startState,
        goalState,
        UCSsuccAndCost,
        ):

        # If a path exists, set |actions| and |totalCost| accordingly.
        # Otherwise, leave them as None.

        self.actions = None
        self.totalCost = None
        self.numStatesExplored = 0

        # Initialize data structures

        frontier = PriorityQueue()  # Explored states are maintained by the frontier.
        backpointers = {}  # map state to (action, previous state)

        # Add the start state

        frontier.update(startState, 0)

        while True:

            # Remove the state from the queue with the lowest pastCost
            # (priority).

            (state, pastCost) = frontier.removeMin()
            if state == None:
                break
            self.numStatesExplored += 1
            if self.verbose >= 2:
                print 'Exploring %s with pastCost %s' % (state,
                        pastCost)

            # Check if we've reached the goal; if so, extract solution

            if state == goalState:
                self.actions = []
                while state != startState:
                    (action, prevState) = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                self.totalCost = pastCost
                if self.verbose >= 1:
                    print 'numStatesExplored = %d' \
                        % self.numStatesExplored
                    print 'totalCost = %s' % self.totalCost
                    print 'actions = %s' % self.actions
                return

            # Expand from |state| to new successor states,
            # updating the frontier with each newState.

            for (action, newState, cost) in UCSsuccAndCost(state):
                if self.verbose >= 3:
                    print '  Action %s => %s with cost %s + %s' \
                        % (action, newState, pastCost, cost)
                if frontier.update(newState, pastCost + cost):

                    # Found better way to go to |newState|, update backpointer.

                    backpointers[newState] = (action, state)
        if self.verbose >= 1:
            print 'No path found'


# Data structure for supporting uniform cost search.

class PriorityQueue:

    def __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  # Map from state to priority

    # Insert |state| into the heap with priority |newPriority| if
    # |state| isn't in the heap or |newPriority| is smaller than the existing
    # priority.
    # Return whether the priority queue was updated.

    def update(self, state, newPriority):
        oldPriority = self.priorities.get(state)
        if oldPriority == None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    # Returns (state with minimum priority, priority)
    # or (None, None) if the priority queue is empty.

    def removeMin(self):
        while len(self.heap) > 0:
            (priority, state) = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE:  # Outdated priority, skip
                continue
            self.priorities[state] = self.DONE
            return (state, priority)
        return (None, None)  # Nothing left...
