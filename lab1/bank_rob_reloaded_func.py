
import numpy as np
import matplotlib.pyplot as plt

import time
import os
import random
from collections import defaultdict

from IPython import display
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from tqdm import tqdm

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED = '#FFC4CC'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'


class City:
    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    BANK1 = (1, 1)

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = 0
    BANK_REWARD = 1
    CAUGHT_REWARD = -10

    def __init__(self, maze, police_cant_stay=True):
        """ Constructor of the environment Maze.
        """
        self.maze = maze
        self.minotaur_cant_stay = police_cant_stay
        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards = self.__rewards()

    def __actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def __states(self):
        states = dict()
        map = dict()
        end = False
        s = 0
        # Player position
        for pi in range(self.maze.shape[0]):
            for pj in range(self.maze.shape[1]):
                # Minotaur position
                for mi in range(self.maze.shape[0]):
                    for mj in range(self.maze.shape[1]):
                        # All combinations of player and minotaur
                        # inside the maze and player not in wall
                        if self.maze[pi, pj] != 1:
                            states[s] = (pi, pj, mi, mj)
                            map[(pi, pj, mi, mj)] = s
                            s += 1
        # NO win or dead like in minotaur
        return states, map

    def __move(self, state, action, for_transition_prob=False):
        """ Player makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the player stays in place.
            Simultaneously the minotaur makes the move

            for_transition_prob --
                returns the len(l) of valid minotaur positions to set t_prob to 1/l
            :return tuple next_state:
                (Px,Py,Mx,My) on the maze that player and minotaur transitions to.
        """
        # For the player
        # Compute the future position given current (state, action)

        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]
        # For minotaur
        # Play a random valid action
        valid_minotaur_moves = self.__minotaur_actions(state, cant_stay=self.minotaur_cant_stay)
        minotaur_pos = random.choice(valid_minotaur_moves)
        # Is the future position an impossible one ?
        hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
                             (col == -1) or (col == self.maze.shape[1]) or \
                             (self.maze[row, col] == 1)
        # Based on the impossiblity check return the next state.

        if for_transition_prob:
            # We can let minotaur take its turn but
            # we would have to handle that in rewards to check if action results in hitting wall
            # Instead of checking if action != stay and position of player remains the same,
            # we could make minotaur also stay so we can just see if the state has changed
            # its just simpler
            if hitting_maze_walls:
                # same state
                return self.states[state][0], self.states[state][1], \
                       [[self.states[state][2], self.states[state][3]]]
            else:
                return row, col, valid_minotaur_moves
        if hitting_maze_walls:
            return self.map[(self.states[state][0], self.states[state][1], minotaur_pos[0], minotaur_pos[1])]
        else:
            return self.map[(row, col, minotaur_pos[0], minotaur_pos[1])]

    def __minotaur_actions(self, state, cant_stay=True):
        # Random action for police
        pos = (self.states[state][2], self.states[state][3])
        valid_moves = []
        actionList = list(self.actions.keys())
        # Get all valid actions for the minotaur position
        if cant_stay and (self.STAY in actionList):
            actionList.remove(self.STAY)
        for action in actionList:
            row = pos[0] + self.actions[action][0]
            col = pos[1] + self.actions[action][1]
            outside_maze = (row == -1) or (row == self.maze.shape[0]) or \
                           (col == -1) or (col == self.maze.shape[1])
            # Assuming minotaur can stay/walk within the walls
            if not outside_maze:
                valid_moves.append([row, col])
        # print("valid_moves", valid_moves)
        return valid_moves

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities.
        # Note that the transitions are probabilistic based on minotaur's random move
        for s in range(self.n_states):
            for a in range(self.n_actions):
                row, col, valid_minotaur_moves = self.__move(s, a, for_transition_prob=True)
                for minotaur_pos in valid_minotaur_moves:
                    next_s = self.map[(row, col, minotaur_pos[0], minotaur_pos[1])]
                    transition_probabilities[next_s, s, a] = 1 / len(valid_minotaur_moves)
        return transition_probabilities

    def __rewards(self):

        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.__move(s, a)

                # Reward for caught by police
                if next_s == s and a != self.STAY:
                    rewards[s][a] = -1000

                elif [self.states[next_s][0], self.states[next_s][1]] == [self.states[next_s][2],
                                                                          self.states[next_s][3]]:
                    rewards[s, a] = self.CAUGHT_REWARD

                # Reward for reaching the bank
                elif (self.states[next_s][0], self.states[next_s][1]) == self.BANK1 and not self.is_dead(next_s):
                    rewards[s, a] = self.BANK_REWARD

                # Reward for taking normal steps
                else:
                    rewards[s, a] = self.STEP_REWARD

        return rewards

    def is_win(self, s):
        return (self.maze[self.states[s][0], self.states[s][1]] == 2) and not self.is_dead(s)

    def is_dead(self, s):
        return self.states[s][0] == self.states[s][2] and self.states[s][1] == self.states[s][3]

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

    def get_next_state(self, state, action):
        return self.__move(state, action)

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            while t < horizon - 1:
                # Get minotaur
                # next_action_m = self.get_minotaur_action(still=False)
                # Move to next state given the policy and the current state
                # next_s = self.__move(s,policy[s,t], next_action_m)
                next_s = self.__move(s, policy[s, t])

                if self.states[next_s][0] == self.states[next_s][2] and self.states[next_s][1] == self.states[next_s][
                    3]:
                    path.append(self.states[next_s])
                    #                     print('Eaten by Minotaur!')
                    return []
                elif self.maze[(self.states[next_s][0], self.states[next_s][1])] == 2:
                    path.append(self.states[next_s])
                    return path

                # Add the position in the maze corresponding to the next state to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
                s = next_s
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Get minotaur
            # next_action_m = self.get_minotaur_action(still=False)
            # Move to next state given the policy and the current state
            # next_s = self.__move(s,policy[s], next_action_m)
            next_s = self.__move(s, policy[s])  # , t

            if self.states[next_s][0] == self.states[next_s][2] and self.states[next_s][1] == self.states[next_s][3]:
                path.append(self.states[next_s])
                # print('Eaten by Minotaur!')
                return path

            # Add the position in the maze corresponding to the next state to the path
            path.append(self.states[next_s])
            # Loop while state is not the goal state
            # while [self.states[s][0], self.states[s][1]] != self.exit:
            while self.maze[(self.states[s][0], self.states[s][1])] != 2:
                # Update state
                s = next_s
                # Get minotaur
                # next_action_m = self.get_minotaur_action(still=False)
                # Move to next state given the policy and the current state
                #                 next_s = self.__move(s,policy[s], next_action_m)
                next_s = self.__move(s, policy[s])
                # Add the position in the maze corresponding to the next state to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
                # print(path)

        return path[:-1]

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


def dynamic_programming(env, horizon):
    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions
    T = horizon

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states, T + 1))
    policy = np.zeros((n_states, T + 1))
    Q = np.zeros((n_states, n_actions))

    # Initialization
    Q = np.copy(r)
    V[:, T] = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming bakwards recursion
    for t in range(T - 1, -1, -1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t + 1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1)
    return V, policy, Q


def value_iteration(env, gamma, epsilon):
    """
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    BV = np.zeros(n_states)
    # Iteration counter
    n = 0
    # Tolerance error
    tol = (1 - gamma) * epsilon / gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # print("Iteration: ", n)
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma * np.dot(p[:, s, a], V)
        BV = np.max(Q, 1)
        # Show error
        # print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q, 1)
    # Return the obtained policy
    return V, policy, Q


def QLearning(env, start, state, steps=10000000):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    r = env.rewards
    n_states = env.n_states
    states = env.states
    n_actions = env.n_actions
    actions = env.actions
    map = env.map
    lambd = 0.8

    Q = np.zeros((n_states, n_actions))
    # number of updates of Q[s, a]
    n = np.zeros((n_states, n_actions))
    start_state = map[start]
    s = start_state
    values_dic = defaultdict(list)
    for i in range(steps):
        if i%1000000 == 0:
            print(i/100000, "% done.")
        a = random.choice(list(actions.keys()))
        next_s = env.get_next_state(s, a)
        alpha = 1 / pow(n[s, a] + 1, 2 / 3)
        Q[s, a] += alpha * (r[s, a] + lambd * max(Q[next_s]) - Q[s, a])

        values_dic[str(state)].append(np.max(Q[state]))
        n[s, a] += 1
        s = next_s

    plt.figure(figsize=(8, 6))
    plt.title("Q-Learning")
    plt.xlabel("Number of Steps")
    plt.ylabel("Value Function")

    for x in values_dic.keys():
        plt.plot(values_dic[x], label=states[int(x)])
    plt.legend(loc=0)
    plt.savefig("Q-learning.png")

    return Q


def SARSA(env, start, state, epsilon_start, steps=10000000):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    r = env.rewards
    n_states = env.n_states
    states = env.states
    n_actions = env.n_actions
    actions = env.actions
    map = env.map
    lambd = 0.8
    Q = np.zeros((n_states, n_actions))
    # number of updates of Q[s, a]
    n = np.zeros((n_states, n_actions))
    start_state = map[start]
    s = start_state
    values_dic_1 = defaultdict(list)
    values_dic = defaultdict(list)

    # epsilon = epsilon
    if epsilon_start == 0.1:
        # Epsilon greedy
        max_a = np.argmax(Q[s])
        action_probs = np.dot([1] * n_actions, epsilon_start / n_actions)
        action_probs[max_a] += 1 - epsilon_start
        a = np.random.choice(list(actions.keys()), p=action_probs)

        for i in range(steps):
            if i % 1000000 == 0:
                print(i / 100000, "% done.")
            next_s = env.get_next_state(s, a)
            alpha = 1 / pow(n[s, a] + 1, 2 / 3)

            # Epsilon greedy
            max_a = np.argmax(Q[s])
            action_probs = np.dot([1] * n_actions, epsilon_start / n_actions)
            action_probs[max_a] += 1 - epsilon_start
            next_a = np.random.choice(list(actions.keys()), p=action_probs)
            Q[s, a] += alpha * (r[s, a] + lambd * Q[next_s][next_a] - Q[s, a])

            values_dic_1[str(state)].append(np.max(Q[state]))

            n[s, a] += 1
            s = next_s
            a = next_a

        plt.figure(figsize=(8, 6))
        plt.title("SARSA Epsilon = 0.1")
        plt.xlabel("Number of Steps")
        plt.ylabel("Value Function")
        for x in values_dic_1.keys():
            plt.plot(values_dic_1[x], label=states[int(x)])
        plt.legend(loc=0)
        plt.savefig("SARSA_epsilon_0.1.png")

    else:
        for epsilon in np.arange(epsilon_start, 0.3, 0.05):
            # Epsilon greedy
            max_a = np.argmax(Q[s])
            action_probs = np.dot([1] * n_actions, epsilon / n_actions)
            action_probs[max_a] += 1 - epsilon
            a = np.random.choice(list(actions.keys()), p=action_probs)

            print("Epsilon: ", epsilon)
            for i in range(steps):
                if i%1000000 == 0:
                    print(i/100000, "% done.")
                next_s = env.get_next_state(s, a)
                alpha = 1 / pow(n[s, a] + 1, 2 / 3)

                # Epsilon greedy
                max_a = np.argmax(Q[s])
                action_probs = np.dot([1] * n_actions, epsilon / n_actions)
                action_probs[max_a] += 1 - epsilon
                next_a = np.random.choice(list(actions.keys()), p=action_probs)
                Q[s, a] += alpha * (r[s, a] + lambd * Q[next_s][next_a] - Q[s, a])

                values_dic[epsilon].append(np.max(Q[state]))

                n[s, a] += 1
                s = next_s
                a = next_a

        plt.figure(figsize=(8, 6))
        plt.title("SARSA")
        plt.xlabel("Number of Steps")
        plt.ylabel("Value Function")
        print(values_dic.keys())
        for x in values_dic.keys():
            plt.plot(values_dic[x], label=f"{x:.2f}")
        plt.legend(loc=0)
        plt.savefig("SARSA.png")

    return Q

def draw_maze(maze):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)


def animate_solution(maze, path):
    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED};

    # Size of the maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0 / rows)
        cell.set_width(1.0 / cols)

    # Update the color at each frame
    for i in range(len(path)):
        grid.get_celld()[(path[i][0], path[i][1])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i][0], path[i][1])].get_text().set_text('Player')
        grid.get_celld()[(path[i][2], path[i][3])].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[(path[i][2], path[i][3])].get_text().set_text('Minotaur')
        if i > 0:
            if maze[path[i][0]][path[i][1]] == 2:  # (, ) == (path[i-1][0], path[i-1][1])
                grid.get_celld()[(path[i][0], path[i][1])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i][0], path[i][1])].get_text().set_text('Player is out')
                grid.get_celld()[(path[i - 1][2], path[i - 1][3])].set_facecolor(
                    col_map[maze[path[i - 1][2], path[i - 1][3]]])
                grid.get_celld()[(path[i - 1][2], path[i - 1][3])].get_text().set_text('')
            else:
                grid.get_celld()[(path[i - 1][2], path[i - 1][3])].set_facecolor(
                    col_map[maze[path[i - 1][2], path[i - 1][3]]])
                grid.get_celld()[(path[i - 1][2], path[i - 1][3])].get_text().set_text('')
                grid.get_celld()[(path[i - 1][0], path[i - 1][1])].set_facecolor(
                    col_map[maze[path[i - 1][0], path[i - 1][1]]])
                grid.get_celld()[(path[i - 1][0], path[i - 1][1])].get_text().set_text('')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)
