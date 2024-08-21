import mdp
from mdp import Action, MDP
from simulator import Simulator
from typing import Dict, List, Tuple
import numpy as np
from copy import deepcopy
import sys


def calc_L_inf_norm(U, U_final, state_x, state_y):
    return np.max(np.abs(U[state_x][state_y] - U_final[state_x][state_y]))


def value_iteration(mdp: MDP, U_init: np.ndarray, epsilon: float=10 ** (-3)) -> np.ndarray:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the utility for each of the MDP's state obtained at the end of the algorithms' run.
    #

    U_final = None
    # TODO:
    # ====== YOUR CODE: ======
    U = np.array(U_init).astype(float)
    U = np.array(U).astype(float)
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

            diff = calc_L_inf_norm(U, U_final, x, y)
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
    policy = np.full((3, 4), None)
    valid_states = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col) if mdp.board[i][j] != 'WALL']
    for x, y in valid_states:
        if (x, y) in mdp.terminal_states:
            continue
        utils_dict = {}
        for action in mdp.actions.keys():
            next_state = mdp.step((x, y), action)
            utils_dict[action] = float(U[next_state[0]][next_state[1]])
        policy[x][y] = max(utils_dict, key=utils_dict.get)
    # ========================
    return policy

def map_matrix_index_to_state(mdp: MDP, index: int):
    x = index // mdp.num_col
    y = index % mdp.num_col
    return x, y

def handle_special_state(index, mdp, R):
    x, y = map_matrix_index_to_state(mdp, index)
    if (x, y) in mdp.terminal_states:
        R[index] = float(mdp.board[x][y])
        return True
    elif mdp.board[x][y] == 'WALL':
        return True
    return False


def derive_key_from_val(dictionary: dict, val):
    for key in dictionary.keys():
        if val == key.value:
            return key

def policy_evaluation(mdp: MDP, policy: np.ndarray) -> np.ndarray:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #
    # TODO:
    # ====== YOUR CODE: ======
    n = mdp.num_col * mdp.num_row
    U = np.zeros(n).astype(float)
    R = np.zeros(n).astype(float)
    I = np.identity(n).astype(float)
    P = np.zeros((n, n)).astype(float)
    for state_idx in range(n):
        x, y = map_matrix_index_to_state(mdp, state_idx)
        if handle_special_state(state_idx, mdp, R):
            continue
        R[state_idx] = float(mdp.board[x][y])
        for next_state_idx in range(n):
            x_next, y_next = map_matrix_index_to_state(mdp, next_state_idx)
            action_by_policy = policy[x][y]
            for index_taken, action_taken in enumerate(mdp.actions.keys()):
                if mdp.step((x, y), action_taken) == (x_next, y_next):
                    # note that the policy say to take action A, but we might take another action.
                    # this is due to non-deterministic transition function. that's why the probability refers to
                    # the transition function at the row of the action A, but with the index of the actual taken action.
                    prob = mdp.transition_function[action_by_policy][index_taken]
                    P[state_idx][next_state_idx] += prob

    U = (np.linalg.inv(I - mdp.gamma * P)) @ R

    return U.reshape(mdp.num_row, mdp.num_col)
    # ========================


def fix_policy_format(mdp, policy):
    for i in range(mdp.num_row):
        for j in range(mdp.num_col):
            if policy[i][j] == None:
                continue
            policy[i][j] = derive_key_from_val(mdp.actions, policy[i][j])

def policy_iteration(mdp: MDP, policy_init: np.ndarray) -> np.ndarray:

    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #
    optimal_policy = None
    # TODO:
    # ====== YOUR CODE: ======
    unchanged = False
    buggy_policy_init = policy_init
    fix_policy_format(mdp, buggy_policy_init)
    optimal_policy = buggy_policy_init
    while not unchanged:
        unchanged = True
        U = policy_evaluation(mdp, optimal_policy)

        valid_states = [(i, j) for i in range(mdp.num_row) for j in range(mdp.num_col) if mdp.board[i][j] != 'WALL']
        for x, y in valid_states:
            if (x, y) in mdp.terminal_states:
                continue
            state = (x, y)
            sum_dict = {}
            action_by_policy = optimal_policy[x][y]

            for action in mdp.actions.keys():
                prob = mdp.transition_function[action]
                sum_lhs = 0.0

                for index, next_state in enumerate(
                        [mdp.step(state, action_taken) for action_taken in mdp.actions.keys()]):
                    next_x, next_y = next_state[0], next_state[1]
                    sum_lhs += prob[index] * (U[next_x][next_y]).item()

                sum_dict[action] = sum_lhs

            sum_rhs = 0.0
            for index, next_state in enumerate(
                    [mdp.step(state, action_taken) for action_taken in mdp.actions.keys()]):
                next_x, next_y = next_state[0], next_state[1]
                probability = mdp.transition_function[action_by_policy][index]
                sum_rhs += probability * (U[next_x][next_y]).item()

            rhs = sum_rhs
            argmax_action = max(sum_dict, key=sum_dict.get)
            lhs = sum_dict[argmax_action]
            if lhs > rhs:
                optimal_policy[x][y] = argmax_action
                unchanged = False

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
    reward_matrix = np.full((num_rows, num_cols), None, dtype=object)
    counter_actions_dict = {Action.UP: 0, Action.DOWN: 0, Action.RIGHT: 0, Action.LEFT: 0}
    counter_actual_actions_dict = {Action.UP: {Action.UP: 0.0, Action.DOWN: 0.0, Action.RIGHT: 0.0, Action.LEFT: 0.0},
                        Action.DOWN: {Action.UP: 0.0, Action.DOWN: 0.0, Action.RIGHT: 0.0, Action.LEFT: 0.0},
                        Action.RIGHT: {Action.UP: 0.0, Action.DOWN: 0.0, Action.RIGHT: 0.0, Action.LEFT: 0.0},
                        Action.LEFT: {Action.UP: 0.0, Action.DOWN: 0.0, Action.RIGHT: 0.0, Action.LEFT: 0.0}}

    transition_probs = counter_actual_actions_dict

    for episode_index, episode_gen in enumerate(sim.replay(num_episodes=num_episodes)):
        for step_index, step in enumerate(episode_gen):
            state, reward, action, actual_action = step
            reward_matrix[state[0]][state[1]] = reward
            if action is not None:
                counter_actions_dict[action] += 1
                counter_actual_actions_dict[action][actual_action] += 1

    # taking care of transition probabilities:
    for possible_action in actions:
        for key in counter_actions_dict.keys():
            transition_probs[possible_action][key] = counter_actual_actions_dict[possible_action][key] / counter_actions_dict[possible_action]
    # ========================
    return reward_matrix, transition_probs 
