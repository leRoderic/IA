# -*- coding: utf-8 -*-
#
# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """Search the deepest nodes in the search tree first."""
    visited, stack = [], util.Stack()
    # The stack starts with the start node.
    stack.push((problem.getStartState(), [], 0))
    while(not stack.isEmpty()):
        pos, dir, cost = stack.pop()
        if problem.isGoalState(pos): # If we've reached the end, the directions to follow will be returned.
            return dir
        if pos not in visited: # If we haven't visited the node, it's added to the visited list.
			visited.append(pos)
            # Also the possible next nodes are added to the stack. Directions and costs are also added to the 
            # ones that they already have.
			for n_pos, n_dir, n_cost in problem.getSuccessors(pos):
				stack.push((n_pos, dir + [n_dir], cost + n_cost))

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    # Same as the DFS algorithm, but instead of using a Stack we'll be using a Queue, so the exploration will
    # start from the shallow nodes before exploring the deepest ones.
    visited, queue = [], util.Queue()
    queue.push((problem.getStartState(), [], 0))
    while(not queue.isEmpty()):
        pos, dir, cost = queue.pop()
        if problem.isGoalState(pos):
            return dir
        if pos not in visited:
			visited.append(pos)
			for n_pos, n_dir, n_cost in problem.getSuccessors(pos):
				queue.push((n_pos, dir + [n_dir], cost + n_cost))

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    visited, queue = [], util.PriorityQueue()
    #entry = (priority, self.count, item)
    # The A* algorithm is based on the previous implementations of the BFS and DFS algorithms.
    # A* uses a priority queue, so that the node with the best heuristic and the least cost is preferred. The
    # heuristic function is passed by parameter.
    queue.push((problem.getStartState(), [], 0), 0)
    while(not queue.isEmpty()):
        pos, dir, cost = queue.pop()
        if problem.isGoalState(pos):
            return dir
        if pos not in visited:
			visited.append(pos)
			for n_pos, n_dir, n_cost in problem.getSuccessors(pos):
            # Path of how to get to the node is added and pushed to the priority queue. The priority is based on the
            # acumulated cost, the new action cost and the heuristic.
				queue.push((n_pos, dir + [n_dir], cost + n_cost), (cost + n_cost + heuristic(n_pos, problem)))

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
