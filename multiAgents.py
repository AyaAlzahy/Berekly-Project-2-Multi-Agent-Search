# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import sys
from util import manhattanDistance
from game import Directions
import random, util
import math
import random
from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition() #next pacman position
        newFood = successorGameState.getFood()  #food position
        newGhostStates = successorGameState.getGhostStates()
        currentGhostStates = currentGameState.getGhostStates()
        "*** YOUR CODE HERE ***"
        numFood = successorGameState.getNumFood() #number of food points
        newGhostPositions = successorGameState.getGhostPositions()
        #get ghost distance
        ghost_distance= float("inf")
        for i in range (len(newGhostPositions)):
            ghost_distance = min(manhattanDistance(newPos,newGhostPositions[i]), ghost_distance)

        #get food distance
        food_distance = float("inf")
        for i in range(newFood.width): #food x coordinate
            for j in range(newFood.height): #food y coordinate
                if newFood[i][j]:
                    temp = abs(manhattanDistance([i,j],newPos))
                    food_distance = min(food_distance,temp)

        return -0.75*numFood+0.5/(food_distance+0.01)-2/(ghost_distance+0.01)


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def minValue(state, depth, agent):
            successor_states = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]
            if len(successor_states) == 0: return self.evaluationFunction(state), None

            minimum = [float("inf"), None] #[value, action]
            for successor_state, action in successor_states:
                 currentValueAction = minMax(successor_state, depth, agent + 1) #[value, action]
                 nextVal= currentValueAction[0]
                 if nextVal < minimum[0]:
                    minimum = [nextVal, action]
            return minimum


        def maxValue(state, depth, agent):
            successor_states = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]
            if len(successor_states) == 0: return self.evaluationFunction(state), None

            maximum = [float("-inf"), None] #[value, action]
            for successor_state, action in successor_states:
                currentValueAction = minMax(successor_state, depth, agent + 1) #[value, action]
                nextVal= currentValueAction[0]
                if nextVal > maximum[0]:
                    maximum = [nextVal, action]

            return maximum

        def minMax(state, depth, agent):
            if agent >= state.getNumAgents(): #last ghost, pacman next
                depth += 1
                agent = 0  #pacman
            if (depth == self.depth):             # Reached max depth
                return [self.evaluationFunction(state), None] #no action
            elif (agent == 0): #pacman
                return maxValue(state, depth, agent)
            else: #ghost
                return minValue(state, depth, agent)
        #get action return
        actions= minMax(gameState, self.index, 0) #actions[0] is the value and actions[1]is the action taken ex:east, west.
        return actions[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
   def getAction(self, state):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def minValue_alphabeta(state, depth, agent, alpha, beta):
            #successor_states = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]
            #if len(successor_states) == 0:  #return self.evaluationFunction(state), None

            minimum = [float("inf"), None] #[value, action]
            legal_actions = state.getLegalActions(agent)

            if len (legal_actions)==0: return self.evaluationFunction(state)

            for action in legal_actions:
                 currState = state.generateSuccessor(agent, action)
                 currentValueAction = minMax_alphabeta(currState, depth, agent + 1, alpha, beta) #[value, action]
                 nextVal= currentValueAction[0]
                 if nextVal < minimum[0]:
                    minimum = [nextVal, action]
                 if nextVal < alpha:
                    #print("min", [nextVal, action])
                    return [nextVal, action]
                 beta = min(beta, nextVal)
            #print("min", minimum)
            return minimum


        def maxValue_alphabeta(state, depth, agent, alpha, beta):
        #    successor_states = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]
        #    if len(successor_states) == 0:       #return [self.evaluationFunction(state), None]

            maximum = [float("-inf"), None] #[value, action]
            legal_actions = state.getLegalActions(agent)

            if len (legal_actions)==0: return self.evaluationFunction(state)

            for action in legal_actions:
                currState = state.generateSuccessor(agent, action)
                currentValueAction = minMax_alphabeta(currState, depth, agent + 1, alpha, beta) #[value, action]
                nextVal= currentValueAction[0]
                if nextVal > maximum[0]:
                    maximum = [nextVal, action]
                if nextVal > beta:
                    #print("max", [nextVal, action])
                    return [nextVal, action]
                alpha = max(alpha, nextVal)
            #print("max", maximum)
            return maximum


        def minMax_alphabeta(state, depth, agent, alpha, beta):
            if agent >= state.getNumAgents(): #last ghost, pacman next
                depth += 1
                agent = 0  #pacman
            #print("depth= ",depth, "self depth= ",self.depth)
            if (depth == self.depth or state.isWin() or state.isLose()):             # Reached max depth
                #print("minOrMax:depth ", self.evaluationFunction(state))
                return [self.evaluationFunction(state), None] #no action
            elif (agent == 0): #pacman
                return maxValue_alphabeta(state, depth, agent, alpha, beta)
            else: #ghost
                return minValue_alphabeta(state, depth, agent, alpha, beta)
        #get action return
        actions= minMax_alphabeta(state, self.index , 0, float("-inf"), float("inf")) #actions[0] is the value and actions[1]is the action taken ex:east, west.
        #print("minOrMax:action list ", actions[1])
        return actions[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def maxValue(state, depth, agent):
            successor_states = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]
            if len(successor_states) == 0: return self.evaluationFunction(state), None

            maximum = [float("-inf"), None] #[value, action]
            for successor_state, action in successor_states:
                currentValueAction = expectiMax(successor_state, depth, agent + 1) #[value, action]
                nextVal= currentValueAction[0]
                if nextVal > maximum[0]:
                    maximum = [nextVal, action]

            return maximum
        def expValue(state, depth, agent):
            successor_states = [(state.generateSuccessor(agent, action), action) for action in state.getLegalActions(agent)]
            if len(successor_states) == 0: return self.evaluationFunction(state), None

            v = [0, None] #[value, action]
            for successor_state, action in successor_states:
                currentValueAction = expectiMax(successor_state, depth, agent + 1) #[value, action]
                v[0] += currentValueAction[0]
            v[0] = v[0]/len(successor_states)
            return v

        def expectiMax(state,depth,agent):
            if agent >= state.getNumAgents(): #last ghost, pacman next
                depth += 1
                agent = 0  #pacman
            if (depth == self.depth):             # Reached max depth
                return [self.evaluationFunction(state), None] #no action
            elif (agent == 0): #pacman
                return maxValue(state, depth, agent)
            else: #ghost
                return expValue(state, depth, agent)
        #get action return
        actions= expectiMax(gameState, self.index, 0)
        return actions[1]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    from heapq import nsmallest
    capsulesPos = currentGameState.getCapsules()
    Pos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    numFood = currentGameState.getNumFood()
    GhostPositions = currentGameState.getGhostPositions()
    ghost_distance= float("inf")
    for i in range (len(GhostPositions)):
        #find the distance between pacman and the closest ghost
        ghost_distance = min(manhattanDistance(Pos,GhostPositions[i]), ghost_distance)
    distances_food =[]
    for i in range(Food.width):
        for j in range(Food.height):
            #find the distance between pacman and all the food pellets
            if Food[i][j]:
                distances_food.append(abs(manhattanDistance([i,j],Pos)))
    distances_food = nsmallest(2,distances_food) #find the distance between pacman and the closest 2 pellets
    distance_food = sum(distances_food)
    distance_capsules = []
    for capsulePos in capsulesPos:
        #find the position of the capsules
        distance_capsules.append(abs(manhattanDistance([i,j],capsulePos)))
    distance_capsules = sum(distance_capsules)
    return -0.75*numFood+0.5/(distance_food+0.01)-2/(ghost_distance+0.01)+10/(distance_capsules+0.01)+random.random()



# Abbreviation
better = betterEvaluationFunction
