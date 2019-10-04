#Artificial Intelligence: A Modern Approach

# Search AIMA
#AIMA Python file: mdp.py

"""Markov Decision Processes (Chapter 17)

First we define an MDP, and the special case of a GridMDP, in which
states are laid out in a 2-dimensional grid.  We also represent a policy
as a dictionary of {state:action} pairs, and a Utility function as a
dictionary of {state:number} pairs.  We then define the value_iteration
and policy_iteration algorithms."""

from utils import *

class MDP:
    """A Markov Decision Process, defined by an initial state, transition model,
    and reward function. We also keep track of a gamma value, for use by
    algorithms. The transition model is represented somewhat differently from
    the text.  Instead of T(s, a, s') being  probability number for each
    state/action/state triplet, we instead have T(s, a) return a list of (p, s')
    pairs.  We also keep track of the possible states, terminal states, and
    actions for each state. [page 615]"""

    def __init__(self, init, actlist, terminals, gamma=.9):
        update(self, init=init, actlist=actlist, terminals=terminals,
               gamma=gamma, states=set(), reward={})

    def R(self, state):
        "Return a numeric reward for this state."
        return self.reward[state]

    def T(state, action):
        """Transition model.  From a state and an action, return a list
        of (result-state, probability) pairs."""
        abstract

    def actions(self, state):
        """Set of actions that can be performed in this state.  By default, a
        fixed list of actions, except for terminal states. Override this
        method if you need to specialize by state."""
        if state in self.terminals:
            return [None]
        else:
            return self.actlist

class GridMDP(MDP):
    """A two-dimensional grid MDP, as in [Figure 17.1].  All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state).  Also, you should specify the terminal states.
    An action is an (x, y) unit vector; e.g. (1, 0) means move east."""
    def __init__(self, grid, terminals, init=(0, 0), gamma=.9):
        grid.reverse() ## because we want row 0 on bottom, not on top
        MDP.__init__(self, init, actlist=orientations,
                     terminals=terminals, gamma=gamma)
        update(self, grid=grid, rows=len(grid), cols=len(grid[0]))
        for x in range(self.cols):
            for y in range(self.rows):
                self.reward[x, y] = grid[y][x]
                if grid[y][x] is not None:
                    self.states.add((x, y))

    def T(self, state, action):
        if action == None:
            return [(0.0, state)]
        else:
            return [(0.8, self.go(state, action)),
                    (0.1, self.go(state, turn_right(action))),
                    (0.1, self.go(state, turn_left(action)))]


    def go(self, state, direction):
        "Return the state that results from going in this direction."
        state1 = vector_add(state, direction)
        return if_(state1 in self.states, state1, state)

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
        return list(reversed([[mapping.get((x,y), None)
                               for x in range(self.cols)]
                              for y in range(self.rows)]))

    def to_arrows(self, policy):
        chars = {(1, 0):'>', (0, 1):'^', (-1, 0):'<', (0, -1):'v', None: '.'}
        return self.to_grid(dict([(s, chars[a]) for (s, a) in policy.items()]))


def value_iteration(mdp, epsilon=0.001):
    "Solving an MDP by value iteration. [Fig. 17.4]"
    U1 = dict([(s, 0) for s in mdp.states])
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    while True:
        U = U1.copy()
        delta = 0
        for s in mdp.states:
            U1[s] = R(s) + gamma * max([sum([p * U[s1] for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon * (1 - gamma) / gamma:
             return U

def best_policy(mdp, U):
    """Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action. (Equation 17.4)"""
    pi = {}
    for s in mdp.states:
        pi[s] = argmax(mdp.actions(s), lambda a:expected_utility(a, s, U, mdp))
    return pi

def expected_utility(a, s, U, mdp):
    "The expected utility of doing a in state s, according to the MDP and U."
    return sum([p * U[s1] for (p, s1) in mdp.T(s, a)])


def policy_iteration(mdp):
    "Solve an MDP by policy iteration [Fig. 17.7]"
    U = dict([(s, 0) for s in mdp.states])
    pi = dict([(s, random.choice(mdp.actions(s))) for s in mdp.states])
    while True:
        U = policy_evaluation(pi, U, mdp)
        unchanged = True
        for s in mdp.states:
            a = argmax(mdp.actions(s), lambda a: expected_utility(a,s,U,mdp))
            if a != pi[s]:
                pi[s] = a
                unchanged = False
        if unchanged:
            return pi

def policy_evaluation(pi, U, mdp, k=20):
    """Return an updated utility mapping U from each state in the MDP to its
    utility, using an approximation (modified policy iteration)."""
    R, T, gamma = mdp.R, mdp.T, mdp.gamma
    for i in range(k):
        for s in mdp.states:
            U[s] = R(s) + gamma * sum([p * U[s] for (p, s1) in T(s, pi[s])])
    return U

#This is the basic graph with given values for rewards. 

myMDP = GridMDP([[0, 0, 0, 0, -1, 0, -1, -1, 0, 50],
                    [None, None, -1, -1, 0, -.5, None, 0, None, 0],
                    [0, 0, 0, 0, 0, -.5, None, 0, 0, 0],
                    [None, 2, None, None, 0, -.5, 0, 2, None, 0],
                    [0, 0, None, 0, 0, None, -1, -.5, -1, 0],
                    [0, -.5, None, 0, 0, None, 0, 0, None, 0],
                    [0, -.5, None, 0, -1, None, 0, -1, None, None],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                    terminals=[(9,7)])

I = value_iteration(myMDP, .001)
print I


'''
Uncomment this section and run to see how the values change for different given values
I decided to ranged these values in increments of .5, feel free to copy and past different 
world on bottom to see more differences

myMDPQ = GridMDP([[-.9, -.9, -.9, -.9, -1, -.9, -1, -1, 0, 75],
                    [None, None, -1, -1, -.9, -.5, None, -.9, None, -.9],
                    [-.9, -.9, -.9, -.9, -.9, -.5, None, -.9, -.9, -.9],
                    [None, 2, None, None, None, -.5, -.9, 2, None, -.9],
                    [None, -.9, -.9, -.9, -.9, None, -1, -.5, -1, -.9],
                    [-.9, -.5, None, -.9, -.9, None, -.9, -.9, None, -.9],
                    [-.9, -.5, None, -.9, -1, None, -.9, -1, None, None],
                    [-.9, -.9, -.9, -.9, -.9, -.9, -.9, -.9, -.9, -.9]],
                    terminals=[(9,7)])

Q = value_iteration(myMDPQ, .001)


myMDPX = GridMDP([[-.5, -.5, -.5, -.5, -1, -.5, -1, -1, 0, 75],
                    [None, None, -1, -1, -.5, -.5, None, -.5, None, -.5],
                    [-.5, -.5, -.5, -.5, -.5, -.5, None, -.5, -.5, -.5],
                    [None, 2, None, None, None, -.5, -.5, 2, None, -.5],
                    [None, -.5, -.5, -.5, -.5, None, -1, -.5, -1, -.5],
                    [-.5, -.5, None, -.5, -.5, None, -.5, -.5, None, -.5],
                    [-.5, -.5, None, -.5, -1, None, -.5, -1, None, None],
                    [-.5, -.5, -.5, -.5, -.5, -.5, -.5, -.5, -.5, -.5]],
                    terminals=[(9,7)])

X = value_iteration(myMDPX, .001)


myMDPY = GridMDP([[-.1, -.1, -.1, -.1, -1, -.1, -1, -1, 0, 75],
                    [None, None, -1, -1, -.1, -.5, None, -.1, None, -.1],
                    [-.1, -.1, -.1, -.1, -.1, -.5, None, -.1, -.1, -.1],
                    [None, 2, None, None, None, -.5, -.1, 2, None, -.1],
                    [None, -.1, -.1, -.1, -.1, None, -1, -.5, -1, -.1],
                    [-.1, -.5, None, -.1, -.1, None, -.1, -.1, None, -.1],
                    [-.1, -.5, None, -.1, -1, None, -.1, -1, None, None],
                    [-.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1]],
                    terminals=[(9,7)])

Y = value_iteration(myMDPY, .001)




myMDP = GridMDP([[0, 0, 0, 0, -1, 0, -1, -1, 0, 50],
                    [None, None, -1, -1, 0, -.5, None, 0, None, 0],
                    [0, 0, 0, 0, 0, -.5, None, 0, 0, 0],
                    [None, 2, None, None, 0, -.5, 0, 2, None, 0],
                    [0, 0, None, 0, 0, None, -1, -.5, -1, 0],
                    [0, -.5, None, 0, 0, None, 0, 0, None, 0],
                    [0, -.5, None, 0, -1, None, 0, -1, None, None],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                    terminals=[(9,7)])

I = value_iteration(myMDP, .001)


myMDPF = GridMDP([[.1, .1, .1, .1, -1, .1, -1, -1, 0, 75],
                    [None, None, -1, -1, .1, -.5, None, .1, None, .1],
                    [.1, .1, .1, .1, .1, -.5, None, .1, .1, .1],
                    [None, 2, None, None, None, -.5, .1, 2, None, .1],
                    [None, .1, .1, .1, .1, None, -1, -.5, -1, .1],
                    [.1, -.5, None, .1, .1, None, .1, .1, None, .1],
                    [.1, -.5, None, .1, -1, None, .1, -1, None, None],
                    [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1]],
                    terminals=[(9,7)])

F = value_iteration(myMDPF, .001)

myMDPZ = GridMDP([[.5, .5, .5, .5, -1, .5, -1, -1, 0, 75],
                    [None, None, -1, -1, .5, -.5, None, .5, None, .5],
                    [.5, .5, .5, .5, .5, -.5, None, .5, .5, .5],
                    [None, 2, None, None, None, -.5, .5, 2, None, .5],
                    [None, .5, .5, .5, .5, None, -1, -.5, -1, .5],
                    [.5, -.5, None, .5, .5, None, .5, .5, None, .5],
                    [.5, -.5, None, .5, -1, None, .5, -1, None, None],
                    [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5]],
                    terminals=[(9,7)])

Z = value_iteration(myMDPZ, .001)


myMDPV = GridMDP([[.9, .9, .9, .9, -1, .9, -1, -1, 0, 75],
                    [None, None, -1, -1, .9, -.5, None, .9, None, .9],
                    [.9, .9, .9, .9, .9, -.5, None, .9, .9, .9],
                    [None, 2, None, None, None, -.5, .9, 2, None, .9],
                    [None, .9, .9, .9, .9, None, -1, -.5, -1, .9],
                    [.9, -.5, None, .9, .9, None, .9, .9, None, .9],
                    [.9, -.5, None, .9, -1, None, .9, -1, None, None],
                    [.9, .9, .9, .9, .9, .9, .9, .9, .9, .9]],
                    terminals=[(9,7)])

V = value_iteration(myMDPV, .001)

print('\n' * 10)
print X
print('\n' * 10)
print Y 
print('\n' * 10)
print I
print('\n' * 10)
print F
print('\n' * 10)
print Z
print('\n' * 10)
print V



Quesiton 1: Terminal states are not anywhere in the matrix, but they are decided as an argument when 
passed into greating the GridMDP at the beginning of the code. The action that is assigned it '= None:'

Question 2: The transistion probablities are defined in 
            def T(self, state, action):
                if action == None:
                    return [(0.0, state)]
                else:
                    return [(0.8, self.go(state, action)),
                            (0.1, self.go(state, turn_right(action))),
                            (0.1, self.go(state, turn_left(action)))]
            where here if we have hit the terminal state there are no more state of actions that need
            to be preformed and we return that position, otherwise we move another spot over, the 
            probabilities of left and rightt are defined as 10 & 10 percent resepectively, with a remaining 80%.
            
Question 3: The fuction that needs to be called to run value_iteration is simply the fucntion itself. You set a value 
equal to the fuction and print out the results of it. In terms of what actually needs to be called, you need the MDP
class and all of the fuctions associated with it like actions, T, R, in order to measure the policy of making move sets 
given the states, then given the Grid class, these fuction are run there are passsed back to value_iteration and what 
it asks for.

Question 4: The utility of running the value_iteration fuction on the MDP providied is:

***************************
(0, 1): 0.3984432178350045 

(3, 1): -1.0, 

(2, 2): 0.7953620878466678 
***************************


**** Leftover values ****

(1, 2): 0.649585681261095 
(3, 2): 1.0
(0, 0): 0.2962883154554812
(3, 0): 0.12987274656746342 
(2, 1): 0.48644001739269643 
(2, 0): 0.3447542300124158 
(1, 0): 0.25386699846479516 
(0, 2): 0.5093943765842497

**************************

Question 5: Actions are represented by a list in th MDP(a list of rewards in the grd). They are measured a number of ways, 
but for the most part the actions are assigned three diffreent ways: in def T, with a return when we we hit terminal(this is 
specified by the user), or a set of results from going in that direction. The fuction def Go is used for the possible
actions and the results of moving in those said directions. 


The data matrix you will need for the assignment given reward changes:


LIVING REWARD: 
    a)  From what I can deduce from this is that given when the non terminal state values are given 0 and 0.5 result in the lowest
        policy iteration values. All values when change result in changes in policy iteration. 

    b)  When changing the values of gamme it showed that the smaller the values of gamma resutled in very very low values 
        output from policy_iteration. Some of these values even dropped into the negative spectrum.
        I didnt ass a way to change it argurment wise, just changed it in the actual function and ran it. 




************Orginal Matrix *******************


[0, 0, 0, 0, -1, 0, -1, -1, 0, 50],
[None, None, -1, -1, 0, -.5, None, 0, None, 0],
[0, 0, 0, 0, 0, -.5, None, 0, 0, 0],
[None, 2, None, None, 0, -.5, 0, 2, None, 0],
[0, 0, None, 0, 0, None, -1, -.5, -1, 0],
[0, -.5, None, 0, 0, None, 0, 0, None, 0],
[0, -.5, None, 0, -1, None, 0, -1, None, None],
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


**********************************************


myMDP = GridMDP([[-0.04, -0.04, -0.04, +1],
                     [-0.04, None,  -0.04, -1],
                     [-0.04, -0.04, -0.04, -0.04]],
                    terminals=[(3,1),(3,2)])


evaluating different rewards:

INITIAL MATRIX:
myMDP = GridMDP([[0, 0, 0, 0, -1, 0, -1, -1, 0, 50],
                    [None, None, -1, -1, 0, -.5, None, 0, None, 0],
                    [0, 0, 0, 0, 0, -.5, None, 0, 0, 0],
                    [None, 2, None, None, 0, -.5, 0, 2, None, 0],
                    [0, 0, None, 0, 0, None, -1, -.5, -1, 0],
                    [0, -.5, None, 0, 0, None, 0, 0, None, 0],
                    [0, -.5, None, 0, -1, None, 0, -1, None, None],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                    terminals=[(9,7)])

MATRICES FROM -1 TO 1:

myMDP = GridMDP([[-.1, -.1, -.1, -.1, -1, -.1, -1, -1, 0, 75],
                    [None, None, -1, -1, -.1, -.5, None, -.1, None, -.1],
                    [-.1, -.1, -.1, -.1, -.1, -.5, None, -.1, -.1, -.1],
                    [None, 2, None, None, None, -.5, -.1, 2, None, -.1],
                    [None, -.1, -.1, -.1, -.1, None, -1, -.5, -1, -.1],
                    [-.1, -.5, None, -.1, -.1, None, -.1, -.1, None, -.1],
                    [-.1, -.5, None, -.1, -1, None, -.1, -1, None, None],
                    [-.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1, -.1]],
                    terminals=[(9,7)])

myMDP = GridMDP([[-.2, -.2, -.2, -.2, -1, -.2, -1, -1, 0, 75],
                    [None, None, -1, -1, -.2, -.5, None, -.2, None, -.2],
                    [-.2, -.2, -.2, -.2, -.2, -.5, None, -.2, -.2, -.2],
                    [None, 2, None, None, None, -.5, -.2, 2, None, -.2],
                    [None, -.2, -.2, -.2, -.2, None, -1, -.5, -1, -.2],
                    [-.2, -.5, None, -.2, -.2, None, -.2, -.2, None, -.2],
                    [-.2, -.5, None, -.2, -1, None, -.2, -1, None, None],
                    [-.2, -.2, -.2, -.2, -.2, -.2, -.2, -.2, -.2, -.2]],
                    terminals=[(9,7)])

myMDP = GridMDP([[-.3, -.3, -.3, -.3, -1, -.3, -1, -1, 0, 75],
                    [None, None, -1, -1, -.3, -.5, None, -.3, None, -.3],
                    [-.3, -.3, -.3, -.3, -.3, -.5, None, -.3, -.3, -.3],
                    [None, 2, None, None, None, -.5, -.3, 2, None, -.3],
                    [None, -.3, -.3, -.3, -.3, None, -1, -.5, -1, -.3],
                    [-.3, -.5, None, -.3, -.3, None, -.3, -.3, None, -.3],
                    [-.3, -.5, None, -.3, -1, None, -.3, -1, None, None],
                    [-.3, -.3, -.3, -.3, -.3, -.3, -.3, -.3, -.3, -.3]],
                    terminals=[(9,7)])

myMDP = GridMDP([[-.4, -.4, -.4, -.4, -1, -.4, -1, -1, 0, 75],
                    [None, None, -1, -1, -.4, -.5, None, -.4, None, -.4],
                    [-.4, -.4, -.4, -.4, -.4, -.5, None, -.4, -.4, -.4],
                    [None, 2, None, None, None, -.5, -.4, 2, None, -.4],
                    [None, -.4, -.4, -.4, -.4, None, -1, -.5, -1, -.4],
                    [-.4, -.5, None, -.4, -.4, None, -.4, -.4, None, -.4],
                    [-.4, -.5, None, -.4, -1, None, -.4, -1, None, None],
                    [-.4, -.4, -.4, -.4, -.4, -.4, -.4, -.4, -.4, -.4]],
                    terminals=[(9,7)])

myMDP = GridMDP([[-.5, -.5, -.5, -.5, -1, -.5, -1, -1, 0, 75],
                    [None, None, -1, -1, -.5, -.5, None, -.5, None, -.5],
                    [-.5, -.5, -.5, -.5, -.5, -.5, None, -.5, -.5, -.5],
                    [None, 2, None, None, None, -.5, -.5, 2, None, -.5],
                    [None, -.5, -.5, -.5, -.5, None, -1, -.5, -1, -.5],
                    [-.5, -.5, None, -.5, -.5, None, -.5, -.5, None, -.5],
                    [-.5, -.5, None, -.5, -1, None, -.5, -1, None, None],
                    [-.5, -.5, -.5, -.5, -.5, -.5, -.5, -.5, -.5, -.5]],
                    terminals=[(9,7)])

myMDP = GridMDP([[-.6, -.6, -.6, -.6, -1, -.6, -1, -1, 0, 75],
                    [None, None, -1, -1, -.6, -.5, None, -.6, None, -.6],
                    [-.6, -.6, -.6, -.6, -.6, -.5, None, -.6, -.6, -.6],
                    [None, 2, None, None, None, -.5, -.6, 2, None, -.6],
                    [None, -.6, -.6, -.6, -.6, None, -1, -.5, -1, -.6],
                    [-.6, -.5, None, -.6, -.6, None, -.6, -.6, None, -.6],
                    [-.6, -.5, None, -.6, -1, None, -.6, -1, None, None],
                    [-.6, -.6, -.6, -.6, -.6, -.6, -.6, -.6, -.6, -.6]],
                    terminals=[(9,7)])

myMDP = GridMDP([[-.7, -.7, -.7, -.7, -1, -.7, -1, -1, 0, 75],
                    [None, None, -1, -1, -.7, -.5, None, -.7, None, -.7],
                    [-.7, -.7, -.7, -.7, -.7, -.5, None, -.7, -.7, -.7],
                    [None, 2, None, None, None, -.5, -.7, 2, None, -.7],
                    [None, -.7, -.7, -.7, -.7, None, -1, -.5, -1, -.7],
                    [-.7, -.5, None, -.7, -.7, None, -.7, -.7, None, -.7],
                    [-.7, -.5, None, -.7, -1, None, -.7, -1, None, None],
                    [-.7, -.7, -.7, -.7, -.7, -.7, -.7, -.7, -.7, -.7]],
                    terminals=[(9,7)])

myMDP = GridMDP([[-.8, -.8, -.8, -.8, -1, -.8, -1, -1, 0, 75],
                    [None, None, -1, -1, -.8, -.5, None, -.8, None, -.8],
                    [-.8, -.8, -.8, -.8, -.8, -.5, None, -.8, -.8, -.8],
                    [None, 2, None, None, None, -.5, -.8, 2, None, -.8],
                    [None, -.8, -.8, -.8, -.8, None, -1, -.5, -1, -.8],
                    [-.8, -.5, None, -.8, -.8, None, -.8, -.8, None, -.8],
                    [-.8, -.5, None, -.8, -1, None, -.8, -1, None, None],
                    [-.8, -.8, -.8, -.8, -.8, -.8, -.8, -.8, -.8, -.8]],
                    terminals=[(9,7)])

myMDP = GridMDP([[-.9, -.9, -.9, -.9, -1, -.9, -1, -1, 0, 75],
                    [None, None, -1, -1, -.9, -.5, None, -.9, None, -.9],
                    [-.9, -.9, -.9, -.9, -.9, -.5, None, -.9, -.9, -.9],
                    [None, 2, None, None, None, -.5, -.9, 2, None, -.9],
                    [None, -.9, -.9, -.9, -.9, None, -1, -.5, -1, -.9],
                    [-.9, -.5, None, -.9, -.9, None, -.9, -.9, None, -.9],
                    [-.9, -.5, None, -.9, -1, None, -.9, -1, None, None],
                    [-.9, -.9, -.9, -.9, -.9, -.9, -.9, -.9, -.9, -.9]],
                    terminals=[(9,7)])

myMDP = GridMDP([[-1, -1, -1, -1, -1, -1, -1, -1, 0, 75],
                    [None, None, -1, -1, -1, -.5, None, -1, None, -1],
                    [-1, -1, -1, -1, -1, -.5, None, -1, -1, -1],
                    [None, 2, None, None, None, -.5, -1, 2, None, -1],
                    [None, -1, -1, -1, -1, None, -1, -.5, -1, -1],
                    [-1, -.5, None, -1, -1, None, -1, -1, None, -1],
                    [-1, -.5, None, -1, -1, None, -1, -1, None, None],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
                    terminals=[(9,7)])

myMDP = GridMDP([[.1, .1, .1, .1, -1, .1, -1, -1, 0, 75],
                    [None, None, -1, -1, .1, -.5, None, .1, None, .1],
                    [.1, .1, .1, .1, .1, -.5, None, .1, .1, .1],
                    [None, 2, None, None, None, -.5, .1, 2, None, .1],
                    [None, .1, .1, .1, .1, None, -1, -.5, -1, .1],
                    [.1, -.5, None, .1, .1, None, .1, .1, None, .1],
                    [.1, -.5, None, .1, -1, None, .1, -1, None, None],
                    [.1, .1, .1, .1, .1, .1, .1, .1, .1, .1]],
                    terminals=[(9,7)])


myMDP = GridMDP([[.2, .2, .2, .2, -1, .2, -1, -1, 0, 75],
                    [None, None, -1, -1, .2, -.5, None, .2, None, .2],
                    [.2, .2, .2, .2, .2, -.5, None, .2, .2, .2],
                    [None, 2, None, None, None, -.5, .2, 2, None, .2],
                    [None, .2, .2, .2, .2, None, -1, -.5, -1, .2],
                    [.2, -.5, None, .2, .2, None, .2, .2, None, .2],
                    [.2, -.5, None, .2, -1, None, .2, -1, None, None],
                    [.2, .2, .2, .2, .2, .2, .2, .2, .2, .2]],
                    terminals=[(9,7)])

myMDP = GridMDP([[.3, .3, .3, .3, -1, .3, -1, -1, 0, 75],
                    [None, None, -1, -1, .3, -.5, None, .3, None, .3],
                    [.3, .3, .3, .3, .3, -.5, None, .3, .3, .3],
                    [None, 2, None, None, None, -.5, .3, 2, None, .3],
                    [None, .3, .3, .3, .3, None, -1, -.5, -1, .3],
                    [.3, -.5, None, .3, .3, None, .3, .3, None, .3],
                    [.3, -.5, None, .3, -1, None, .3, -1, None, None],
                    [.3, .3, .3, .3, .3, .3, .3, .3, .3, .3]],
                    terminals=[(9,7)])


myMDP = GridMDP([[.4, .4, .4, .4, -1, .4, -1, -1, 0, 75],
                    [None, None, -1, -1, .4, -.5, None, .4, None, .4],
                    [.4, .4, .4, .4, .4, -.5, None, .4, .4, .4],
                    [None, 2, None, None, None, -.5, .4, 2, None, .4],
                    [None, .4, .4, .4, .4, None, -1, -.5, -1, .4],
                    [.4, -.5, None, .4, .4, None, .4, .4, None, .4],
                    [.4, -.5, None, .4, -1, None, .4, -1, None, None],
                    [.4, .4, .4, .4, .4, .4, .4, .4, .4, .4]],
                    terminals=[(9,7)])


myMDP = GridMDP([[.5, .5, .5, .5, -1, .5, -1, -1, 0, 75],
                    [None, None, -1, -1, .5, -.5, None, .5, None, .5],
                    [.5, .5, .5, .5, .5, -.5, None, .5, .5, .5],
                    [None, 2, None, None, None, -.5, .5, 2, None, .5],
                    [None, .5, .5, .5, .5, None, -1, -.5, -1, .5],
                    [.5, -.5, None, .5, .5, None, .5, .5, None, .5],
                    [.5, -.5, None, .5, -1, None, .5, -1, None, None],
                    [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5]],
                    terminals=[(9,7)])

myMDP = GridMDP([[.6, .6, .6, .6, -1, .6, -1, -1, 0, 75],
                    [None, None, -1, -1, .6, -.5, None, .6, None, .6],
                    [.6, .6, .6, .6, .6, -.5, None, .6, .6, .6],
                    [None, 2, None, None, None, -.5, .6, 2, None, .6],
                    [None, .6, .6, .6, .6, None, -1, -.5, -1, .6],
                    [.6, -.5, None, .6, .6, None, .6, .6, None, .6],
                    [.6, -.5, None, .6, -1, None, .6, -1, None, None],
                    [.6, .6, .6, .6, .6, .6, .6, .6, .6, .6]],
                    terminals=[(9,7)])

myMDP = GridMDP([[.7, .7, .7, .7, -1, .7, -1, -1, 0, 75],
                    [None, None, -1, -1, .7, -.5, None, .7, None, .7],
                    [.7, .7, .7, .7, .7, -.5, None, .7, .7, .7],
                    [None, 2, None, None, None, -.5, .7, 2, None, .7],
                    [None, .7, .7, .7, .7, None, -1, -.5, -1, .7],
                    [.7, -.5, None, .7, .7, None, .7, .7, None, .7],
                    [.7, -.5, None, .7, -1, None, .7, -1, None, None],
                    [.7, .7, .7, .7, .7, .7, .7, .7, .7, .7]],
                    terminals=[(9,7)])

myMDP = GridMDP([[.8, .8, .8, .8, -1, .8, -1, -1, 0, 75],
                    [None, None, -1, -1, .8, -.5, None, .8, None, .8],
                    [.8, .8, .8, .8, .8, -.5, None, .8, .8, .8],
                    [None, 2, None, None, None, -.5, .8, 2, None, .8],
                    [None, .8, .8, .8, .8, None, -1, -.5, -1, .8],
                    [.8, -.5, None, .8, .8, None, .8, .8, None, .8],
                    [.8, -.5, None, .8, -1, None, .8, -1, None, None],
                    [.8, .8, .8, .8, .8, .8, .8, .8, .8, .8]],
                    terminals=[(9,7)])

myMDP = GridMDP([[.9, .9, .9, .9, -1, .9, -1, -1, 0, 75],
                    [None, None, -1, -1, .9, -.5, None, .9, None, .9],
                    [.9, .9, .9, .9, .9, -.5, None, .9, .9, .9],
                    [None, 2, None, None, None, -.5, .9, 2, None, .9],
                    [None, .9, .9, .9, .9, None, -1, -.5, -1, .9],
                    [.9, -.5, None, .9, .9, None, .9, .9, None, .9],
                    [.9, -.5, None, .9, -1, None, .9, -1, None, None],
                    [.9, .9, .9, .9, .9, .9, .9, .9, .9, .9]],
                    terminals=[(9,7)])

myMDP = GridMDP([[1, 1, 1, 1, -1, 1, -1, -1, 0, 75],
                    [None, None, -1, -1, 1, -.5, None, 1, None, 1],
                    [1, 1, 1, 1, 1, -.5, None, 1, 1, 1],
                    [None, 2, None, None, None, -.5, 1, 2, None, 1],
                    [None, 1, 1, 1, 1, None, -1, -.5, -1, 1],
                    [1, -.5, None, 1, 1, None, 1, 1, None, 1],
                    [1, -.5, None, 1, -1, None, 1, -1, None, None],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
                    terminals=[(9,7)])

implmenteding the horse jump-stuff

    def T(self, state, action):
        jumps = [(2.0, (-2.0), (0,2), (0,-2)]
        if action == None:
            return [(0.0, state)]
        else:
            if action in jumps:
                return [0.5, self.go(state, action)], (0.5, state)]
            else:
                return [(0.8, self.go(state, action)),
                        (0.1, self.go(state, turn_right(action))),
                        (0.1, self.go(state, turn_left(action)))]

Purpose: 
Describe the purpose of the assignment and what you want to  accomplish.

The purpose of this assignment was to see the changes in policy iteration and how 
different numbers of rewards values change the final utility. Here we see how gamma 
and 



Procedure:  Your procedure section needs to include the modifications you made to  
the code and the experiments you ran, including the range of values used in each    
experiment. 

This is all shown above in my large comments section and what values I used and how I changed gamma, the 
new reward values in the non-terminal states and implmented the horse jumping when. 


Data: Describe the data, including specific values in the matrix and what they    
represent.

The data was all output in the terminal, its sloppy i know but i basically printed out all the 
different possibilites of the graphes and the values of their states given the policy iteration 
function and how the reward and gamma values change the policy. 


Results: Describe the results of the experiments modifying the living reward,  
gamma, actions, and transition probabilities for the jumping action.

The results all changed when experimenting on the graph. The most drastic change was changing the gamma value. For the most
part the values seemed to be in the range of 20-35 when other changes were made. But when the gamma change was introduced 
these values shot down to the single digits and even negative values. 

'''