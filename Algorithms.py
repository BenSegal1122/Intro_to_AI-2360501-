from os import close
import numpy as np

from CampusEnv import CampusEnv
from typing import List, Tuple
import heapdict


class DFSGAgent:
    def __init__(self) -> None:
        self.solution_cost = 0
        self.solution_actions = []
        self.expanded_nodes = 0

    def search(self, env: CampusEnv):
        node = self.make_node(0, 1)
        print(node)
        open_group = []
        close_group = []
        cost = 0
        actions = []
        open_group.append(node)
        self.recursive_dfs(env, open_group, close_group, cost, actions)
        return actions, cost, expanded_nodes

    def make_node(self, state, cost):
        created_node = {"state": state, "cost": cost}
        return created_node

    def expand(self, st, env):
        neighbors = {}
        neighbors = env.succ(st)  # TODO: check what order
        return neighbors
        # neighbors = {"Down": (new state, cost, terminated),
        # "UP": (new state, cost, terminated), ... }

    def recursive_dfs(self, env, open_group, close_group, cost, actions):
        current_node = open_group.pop(0)
        close_group.append(current_node["state"])
        st = current_node["state"]
        if env.is_final_state(st):
            return
        for key, value in self.expand(current_node["state"], env).items():
            child = self.make_node(value[0], value[1])
            if child["state"] not in close_group and child not in open_group:
                open_group.append(child)
                actions.append(key)
                result = self.recursive_dfs(env, open_group, close_group, cost, actions)
                if result != "failed": return "Success"
                open_group.pop()
                actions.pop()
        return "failed"


class UCSAgent():

    def __init__(self) -> None:
        raise NotImplementedError

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError


class WeightedAStarAgent():

    def __init__(self):
        raise NotImplementedError

    def search(self, env: CampusEnv, h_weight) -> Tuple[List[int], float, int]:
        raise NotImplementedError


class AStarAgent():

    def __init__(self):
        raise NotImplementedError

    def search(self, env: CampusEnv) -> Tuple[List[int], float, int]:
        raise NotImplementedError

