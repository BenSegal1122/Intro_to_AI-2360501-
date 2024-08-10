from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np
from copy import deepcopy
import sys


def calc_L1_norm(U, U_final, state_x, state_y):
    return abs(U[state_x][state_y].item() - U_final[state_x][state_y].item())

###################################################################


####################################################################
def value_iteration3(mdp: MDP, U_init: np.ndarray, epsilon: float = 10 ** (-3)) -> np.ndarray:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    #

    U_final = None
    # TODO:
    # ====== YOUR CODE: ======
    U = U_init.astype(float)
    while True:
        U_final = U.astype(float)
        delta = 0
        valid_states = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col) if mdp.board[i][j] != 'WALL']
        for x, y in valid_states:
            state = (x, y)
            if state in mdp.terminal_states:
                U[x][y] = float(mdp.board[x][y])
            else:
                max_util = None
                for action in mdp.actions.keys():
                    prob = mdp.transition_function[action]
                    util_list = []
                    for next_state in [mdp.step(state, action_taken) for action_taken in mdp.actions.keys()]:
                        next_x, next_y = next_state[0], next_state[1]
                        util_list.append((U[next_x][next_y]).item())
                    curr_utility = sum(np.array(prob) * np.array(util_list))

                    if not max_util or max_util < curr_utility:
                        max_util = curr_utility

                U[x][y] = float(mdp.board[x][y]) + mdp.gamma * max_util

            delta = max(delta, abs(U[x][y].item() - U_final[x][y].item()))
            if delta < ((epsilon * (1 - mdp.gamma)) / mdp.gamma):
                return U_final

    # ========================
    return U_final


####################################################################
def value_iteration(mdp: MDP, U_init: np.ndarray, epsilon: float=10 ** (-3)) -> np.ndarray:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    #

    U_final = None
    # TODO:
    # ====== YOUR CODE: ======
    U = U_init.astype(float)
    while True:
        U_final = U.astype(float)
        delta = 0
        valid_states = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col) if mdp.board[i][j] != 'WALL']
        for x, y in valid_states:
            state = (x, y)
            if state in mdp.terminal_states:
                U[x][y] = float(mdp.board[x][y])
            else:
                sum_list = []
                for action in mdp.actions.keys():
                    prob = mdp.transition_function[action]
                    sum = 0.0
                    for index, next_state in enumerate(
                            [mdp.step(state, action_taken) for action_taken in mdp.actions.keys()]):
                        next_x, next_y = next_state[0], next_state[1]
                        sum += prob[index] * (U[next_x][next_y]).item()
                    sum_list.append(sum)
                U[x][y] = float(mdp.board[x][y]) + mdp.gamma * max(sum_list)

            diff = calc_L1_norm(U, U_final, x, y)
            if delta < diff:
                delta = diff
            if delta < ((epsilon * (1 - mdp.gamma)) / mdp.gamma):
                return U_final

    # ========================
    return U_final

def get_policy(mdp: MDP, U: np.ndarray) -> np.ndarray:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #
    
    policy = None
    # TODO:
    # ====== YOUR CODE: ====== 
    raise NotImplementedError
    # ========================
    return policy


def policy_evaluation(mdp: MDP, policy: np.ndarray) -> np.ndarray:

    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    # TODO:
    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp: MDP, policy_init: np.ndarray) -> np.ndarray:

    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = None
    # TODO:
    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
    return optimal_policy



def adp_algorithm(
    sim: Simulator, 
    num_episodes: int,
    num_rows: int = 3, 
    num_cols: int = 4, 
    actions: List[Action] = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT] 
) -> Tuple[np.ndarray, Dict[Action, Dict[Action, float]]]:
    """
    Runs the ADP algorithm given the simulator, the number of rows and columns in the grid, 
    the list of actions, and the number of episodes.

    :param sim: The simulator instance.
    :param num_rows: Number of rows in the grid (default is 3).
    :param num_cols: Number of columns in the grid (default is 4).
    :param actions: List of possible actions (default is [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]).
    :param num_episodes: Number of episodes to run the simulation (default is 10).
    :return: A tuple containing the reward matrix and the transition probabilities.
    
    NOTE: the transition probabilities should be represented as a dictionary of dictionaries, so that given a desired action (the first key),
    its nested dicionary will contain the condional probabilites of all the actions. 
    """
    

    transition_probs = None
    reward_matrix = None
    # TODO
    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
    return reward_matrix, transition_probs 
