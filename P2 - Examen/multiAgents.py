# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

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
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        # Using Manhattan distance, the distance to each ghost is computed but only the smallest one is saved.
        # That's because the closest one is the most dangerous one.
        closestGhost = min([manhattanDistance(newPos, g.getPosition()) for g in newGhostStates if g.scaredTimer == 0])
        
        # If the new position is close to the closest ghost, the score will be really low in order to avoid
        # getting pacman eaten by a ghost.
        if closestGhost < 2:
            return -9999999
        
        # Since the game is won after all pellets are eaten, being near them is encouraged. The distance to the 
        # furthest pellet is computed and given as score. This way succesors who have more food near them will 
        # be chosen.
        return max(-manhattanDistance(food, newPos) for food in currentGameState.getFood().asList())

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

        # Minimax function is called for agent=0 (pacman). The function returns both action and action_score,
        # but since 'getAction' only returns the action, the score is ignored.
        _, action = self.minimax(gameState, self.index, 0)
        return action
        
    def minimax(self, gameState, agentIndex, depth):
            
        # Once the agents turn has passed (minimized), it's pacman turn again (maximize) so the index and depth
        # are updated.
        if agentIndex >= gameState.getNumAgents():
        
            agentIndex = 0
            depth -= -1
            
        actions = gameState.getLegalActions(agentIndex)
        
        # When a 'terminal node' is reached or there are not legal actions available, the evaluationFunction implemented
        # on the first exercise is used as an utility function.
        if depth == self.depth or not actions:
            return self.evaluationFunction(gameState)
        
        # Pacman looks after maximizing its gain, so it will select the maximum scored action. At the same time, will try
        # to minimize the ghosts' gain.
        if agentIndex == 0:
            return max([(self.minimax(gameState.generateSuccessor(agentIndex, i), agentIndex + 1, depth), i) for i in actions])
        else:
            return min([(self.minimax(gameState.generateSuccessor(agentIndex, i), agentIndex + 1, depth), i) for i in actions])
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Same as the minimax call, except alpha and beta values on first call are -infinity and infinity respectively.
        _, action = self.alphabeta_pruning(gameState, self.index, 0, float("-inf"), float("inf"))
        return action
    
    def alphabeta_pruning(self, gameState, agentIndex, depth, alpha, beta):
        
        # Algorithm similar to minimax, only changes/additions are explained.
        if agentIndex >= gameState.getNumAgents():
        
            agentIndex = 0
            depth -= -1
            
        actions = gameState.getLegalActions(agentIndex)
        
        if depth == self.depth or not actions:
            return self.evaluationFunction(gameState), None
        
        if agentIndex == 0:
            
            # Initial values are -infinity and action 'None'.
            value, action = float("-inf"), None
            # A simple list comprehension as in the minimax algorithm cannot be used here, since there are a lot
            # of multiple assignations and returns in the middle.
            for i in actions:
                
                nvalue, naction = self.alphabeta_pruning(gameState.generateSuccessor(agentIndex, i), agentIndex + 1, depth, alpha, beta)
                
                # If the value has improved, it is kept. Here we're maximizing.
                if nvalue > value:
                    value, action = nvalue, i
                
                # Pruning when value is bigger than beta.
                if value > beta:
                    return value, action
                
                # Alpha will be always the maximum, so it's compared and updated with the new values.
                alpha = max(alpha, value)
            
            # If no pruning has occured, last values are returned.
            return value, action
        else:
        # Initial values are infinity and action 'None'.
            value, action = float("inf"), None
            for i in actions:
                
                nvalue, naction = self.alphabeta_pruning(gameState.generateSuccessor(agentIndex, i), agentIndex + 1, depth, alpha, beta)
                
                # If the value has improved, it is kept. Here we're minimizing.
                if nvalue < value:
                    value, action = nvalue, i
                
                # Pruning when value is smaller than alpha.
                if value < alpha:
                    return value, action
                
                # Beta will be always the minimum, so it's compared and updated with the new values.
                beta = min(beta, value)
            
            # If no pruning has occured, last values are returned.
            return value, action
    
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
        # Same as the minimax call. The algorithm changes though.
        _, action = self.expectimax(gameState, self.index, 0)
        return action
    
    def expectimax(self, gameState, agentIndex, depth):
        
        if agentIndex >= gameState.getNumAgents():
        
            agentIndex = 0
            depth -= -1
        
        actions = gameState.getLegalActions(agentIndex)
        
        if depth == self.depth or not actions:
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            # The maximum gain will be the goal for agent Pacman. To achieve this, the maximum scored action will be 
            # returned.
            value, action = float("-inf"), None
            for i in actions:
                nvalue, naction = self.expectimax(gameState.generateSuccessor(agentIndex, i), agentIndex + 1, depth)
                
                if nvalue > value:
                    value, action = nvalue, i
            
            return value, action
        else:
            # Ghosts here have a suboptimal behavior, what will imply a probability and a random chosen action. The
            # average of all 'scores' will be used as probability and a random action will be selected using the choice
            # function from the 'random' library.
            values = [self.expectimax(gameState.generateSuccessor(agentIndex, i), agentIndex + 1, depth)[0] for i in actions]
            # Returning the average it's the same as multiplying each score by it's probability (1/number_of_actions).
            return float(sum(values))/len(actions), random.choice(actions)

class expectimaxMinimaxAgent2(MultiAgentSearchAgent):
    """
      P2 exam agent.
      
      python pacman.py -p expectimaxMinimaxAgent2 -l minimaxClassic -a depth=3
    """

    def getAction(self, gameState):

        _, action = self.expectimax(gameState, self.index, 0)
        return action
    
    def expectimax(self, gameState, agentIndex, depth):
        
        if agentIndex >= gameState.getNumAgents():
        
            agentIndex = 0
            depth -= -1
        
        actions = gameState.getLegalActions(agentIndex)
        
        if depth == self.depth or not actions:
            return self.evaluationFunction(gameState), None

        if agentIndex == 0:
            
            value, action = float("-inf"), None
            for i in actions:
                nvalue, naction = self.expectimax(gameState.generateSuccessor(agentIndex, i), agentIndex + 1, depth)
                
                if nvalue > value:
                    value, action = nvalue, i
            
            return value, action
        
        elif agentIndex%2!=0:
            # Odd index agents.
            # I.e.first, third, fifth, ...
            value, action = float("inf"), None
            for i in actions:
                nvalue, naction = self.expectimax(gameState.generateSuccessor(agentIndex, i), agentIndex + 1, depth)
                # Minimax behavior, keeps the minimum action-cost.
                if nvalue < value:
                    value, action = nvalue, i

            return value, action
        else: 
            # Even index agents.
            # I.e. second, fourth, sixth, ...
            value, action = float("-inf"), None
            # If there are more than one unique legal action, the action with the highest value will be removed
            if len(actions) > 1:
                for i in actions:
                    nvalue, naction = self.expectimax(gameState.generateSuccessor(agentIndex, i), agentIndex + 1, depth)
                    
                    if nvalue > value:
                        value, action = nvalue, i
                # Once the minimum-cost action has been found, it is removed from the legal actions list.
                actions.remove(action)
        
            values = [self.expectimax(gameState.generateSuccessor(agentIndex, i), agentIndex + 1, depth)[0] for i in actions]
            
            return float(sum(values))/len(actions), random.choice(actions)
            
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: The enhanced evaluation function now takes into account, remaining pellets as well as remaining
      capsules, distance to closest ghost and closest pellet, power time (where it Pacman can eat any ghost) and 
      new measure: shouldiness attack coefficient.  This latter one indicates if pacman should go after the ghosts
      or if it should remain to eating pellets and running away instead.
    """
    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood().asList()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    # Amount of pellets and capsules that haven't been eaten yet.
    notEatenFood = len(newFood) + len(currentGameState.getCapsules())
    # Distance to closest ghost, the most dangerous one.
    closestGhost = min([manhattanDistance(newPos, g.getPosition()) for g in newGhostStates])
    # Distance to closest pellet. It sets a 0 value when there isn't any pending food to avoid errors.
    closestPellet = min([manhattanDistance(newPos, p) for p in newFood]) if newFood else 0
    # Remaining time of 'attack' in the worst case scenario, i.e. the minimum.
    killingTime = min(newScaredTimes)
    # Shouldiness Attack Coefficient. It encourages attack when pacman is powerful (can kill ghosts) and discourages it
    # when killingTime is 0.
    ghostEater = -2*closestGhost if not killingTime else 0.5*closestGhost
    
    # 1 is added to closePellet feature division to avoid division by 0 errors.
    return currentGameState.getScore() - notEatenFood + ghostEater + 0.5/(closestPellet + 1) + killingTime
    
# Abbreviation
better = betterEvaluationFunction

