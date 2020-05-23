# -*- coding: utf-8 -*-
# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        # Using Counter to keep track of the Q-Values. Using Counter from 'util.py' ensures that no errors
        # are produced when accesing a non-existent key.
        self.qValues = util.Counter()

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        # Since the Counter already returns 0.0 for non-existing keys, only the access to the 'dictionary' is performed
        # and then returned.
        return self.qValues[(state, action)]


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        actions = self.getLegalActions(state)
        
        # If a terminal state is reached, the returned Q-Value will be 0.0.
        if not actions: return 0.0
        
        # Otherwise, the maximum Q-Value from all the actions will be returned.
        return max([self.getQValue(state, i) for i in actions])

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        actions = self.getLegalActions(state)
        
        # Once a terminal state is reached there are no more possible actions, so 'None' is returned.
        if not actions: return None
        
        # Similar to 'computeValueFromQValues'. A list of (Q-Value, action) is created and sorted in decreasing order
        # by its first index (Q-Values). This leaves the action with the maximum Q-Value on the first index and is later
        # returned.
        _, ract = sorted([(self.getQValue(state, i), i) for i in actions], key=lambda x: x[0], reverse=True)[0]
        
        return ract #random.choice(actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        actions = self.getLegalActions(state)
        
        # Using 'flipCoin' function from util.py, which returns True or False with a determined probability, the Epsilon
        # Greedy action selection (Q2) is implemented. It will choose a random action with a probability of epsilon, and
        # will choose the best action with a probability of (1 - epsilon).
        action = random.choice(actions) if util.flipCoin(self.epsilon) else self.computeActionFromQValues(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        actions = self.getLegalActions(nextState)
        
        # If a terminal state is reached, only the reward will be taken into account since no legal actions will be
        # avaible.
        sample = reward
        
        # Otherwise the maximum Q-Value from the next possible legal actions will be included weighted by the discount
        # factor, which determines how much the Agent cares about future rewards
        if actions: sample += self.discount*max([self.getQValue(nextState, i) for i in actions]) 
        
        # The new estimation is saved.
        self.qValues[(state, action)] = (1 - self.alpha)*self.getQValue(state, action) + self.alpha*sample

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action
