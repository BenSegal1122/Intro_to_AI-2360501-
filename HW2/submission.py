from Agent import Agent, AgentGreedy
from WarehouseEnv import WarehouseEnv, manhattan_distance
import random
import time
import math

safety_epsilon = 1e-2

def manhatan_dist(source, destination):
    return abs(source[0] - destination[0]) + abs(source[1] - destination[1])

# TODO: section a : 3
# def smart_heuristic(env: WarehouseEnv, robot_id: int):
    # robot = env.robots[robot_id]
    # has_package = robot.package is not None
    # if not has_package:
    #     closest_package_index = min([0, 1], key=lambda i: manhatan_dist(robot.position, env.packages[i].position))
    #     closest_package = env.packages[closest_package_index]
    #     closest_package_credit = manhatan_dist(closest_package.position, closest_package.destination)
    #     if manhatan_dist(robot.position, closest_package.position) < robot.battery:
    #         return (manhatan_dist(robot.position, closest_package.position) *
    #                 (robot.battery - 2 * closest_package_credit))
    #
    #     else:
    #         closest_charge_index = min([0, 1], key=lambda i: manhatan_dist(robot.position, env.charge_stations[i].position))
    #         closest_charge_station = env.packages[closest_charge_index]
    #         return (manhatan_dist(robot.position, closest_charge_station.position) *
    #                 (robot.battery - closest_package_credit))
    # else:
    #     package_destination = robot.package.destination
    #     package_source = robot.package.position
    #     package_credit = manhatan_dist(package_destination, package_source)
    #     if manhatan_dist(robot.position, package_destination) < robot.battery:
    #         return (manhatan_dist(robot.position, package_destination) *
    #                 (robot.battery - 2 * package_credit))
    #
    #     else:
    #         closest_charge_index = min([0, 1], key=lambda i: manhatan_dist(robot.position, env.charge_stations[i].position))
    #         closest_charge_station = env.packages[closest_charge_index]
    #         return (manhatan_dist(robot.position, closest_charge_station.position) *
    #                 (robot.battery - package_credit))

def closest_charge_station(env: WarehouseEnv, pos):
    curr_closest_station = math.inf
    for station in env.charge_stations:
        distance = manhattan_distance(pos, station.position)
        if distance < curr_closest_station:
            curr_closest_station = distance
    return curr_closest_station


def closest_package(env: WarehouseEnv, pos):
    curr_closest_package = math.inf
    chosen_package = None
    for package in env.packages:
        distance = manhattan_distance(pos, package.position)
        if package.on_board and distance < curr_closest_package:
            curr_closest_package = distance
            chosen_package = package
    return curr_closest_package, chosen_package


def package_worth(package):
    if package:
        return 2*manhattan_distance(package.position, package.destination)
    return 0


def dis_to_destination(env: WarehouseEnv, robot):
    if robot.package:
        return manhattan_distance(robot.position, robot.package.destination)
    return math.inf


# section a : 3
def smart_heuristic(env: WarehouseEnv, robot_id: int):
    robot = env.get_robot(robot_id)
    if robot.package:
        if robot.battery > dis_to_destination(env, robot):
            return (3 * robot.battery - 6 * dis_to_destination(env, robot) +
                    6 * max(robot.credit, package_worth(robot.package)))
        else:
            return 3*robot.battery - closest_charge_station(env, robot.position) + robot.credit
    else:
        closest_package_distance, package = closest_package(env, robot.position)
        if robot.battery > closest_package_distance:
            return (3*robot.battery - 3*closest_package_distance) + (6*robot.credit)
        else:
            return (3*robot.battery - closest_charge_station(env, robot.position)) + 2*robot.credit

class AgentGreedyImproved(AgentGreedy):
    def heuristic(self, env: WarehouseEnv, robot_id: int):
        return smart_heuristic(env, robot_id)


class AgentMinimax(Agent):
    # TODO: section b : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        depth = 0
        best_op = None
        while True:
            try:
                ops, children = self.successors(env, agent_id)
                next_turn = abs(agent_id - 1)
                heuristic_opts = [self.RB_minimax(child, agent_id, time_limit, start_time, depth, next_turn) for child in children]
                best_val = max(heuristic_opts)
                best_ops_list = [index for index, h_val in enumerate(heuristic_opts) if h_val == best_val]
                best_op = ops[random.choice(best_ops_list)]
                depth += 1
            except TimeoutError:
                return best_op


    def RB_minimax(self, env: WarehouseEnv, agent_id, time_limit, start_time, depth, turn):
        now = time.time()
        if now - start_time > time_limit - safety_epsilon:
            raise TimeoutError
        if depth == 0 or env.done():
            return smart_heuristic(env, agent_id)

        ops, children = self.successors(env, agent_id)
        if turn == agent_id: # max case
            curr_max = -(math.inf)
            for op, child in zip(ops, children):
                v = self.RB_minimax(child, agent_id, time_limit, start_time, depth - 1, abs(turn - 1))
                curr_max = max(curr_max, v)
            return curr_max

        else: # min case
            curr_min = math.inf
            for op, child in zip(ops, children):
                v = self.RB_minimax(child, agent_id, time_limit, start_time, depth - 1, abs(turn - 1))
                curr_min = min(curr_min, v)
            return curr_min



class AgentAlphaBeta(Agent):
    # TODO: section c : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        depth = 0
        best_op = None
        alpha = -(math.inf)
        beta = math.inf
        while True:
            try:
                ops, children = self.successors(env, agent_id)
                next_turn = abs(agent_id - 1)
                heuristic_opts = [self.RB_AlphaBeta(child, agent_id, time_limit, start_time, depth, next_turn, alpha, beta) for child in
                                  children]
                best_val = max(heuristic_opts)
                best_ops_list = [index for index, h_val in enumerate(heuristic_opts) if h_val == best_val]
                best_op = ops[random.choice(best_ops_list)]
                depth += 1
            except TimeoutError:
                return best_op

    def RB_AlphaBeta(self, env: WarehouseEnv, agent_id, time_limit, start_time, depth, turn, alpha, beta):
        now = time.time()
        if now - start_time > time_limit - safety_epsilon:
            raise TimeoutError
        if depth == 0 or env.done():
            return smart_heuristic(env, agent_id)

        ops, children = self.successors(env, agent_id)
        if turn == agent_id:  # max case
            curr_min = -(math.inf)
            for child in children:
                v = self.RB_AlphaBeta(child, agent_id, time_limit, start_time, depth - 1, abs(turn - 1), alpha, beta)
                curr_max = max(curr_max, v)
                alpha = max(curr_max, alpha)
                if curr_max >= beta:
                    return math.inf
            return curr_max

        else: # min case
            curr_min = math.inf
            for child in children:
                v = self.RB_AlphaBeta(child, agent_id, time_limit, start_time, depth - 1, abs(turn - 1), alpha, beta)
                curr_min = min(curr_min, v)
                beta = min(curr_min, beta)
                if curr_min <= alpha:
                    return -(math.inf)
            return curr_min


class AgentExpectimax(Agent):
    # TODO: section d : 1
    def run_step(self, env: WarehouseEnv, agent_id, time_limit):
        start_time = time.time()
        depth = 0
        best_op = None
        while True:
            try:
                ops, children = self.successors(env, agent_id)
                next_turn = abs(agent_id - 1)
                heuristic_opts = [self.RB_expectimax(child, agent_id, time_limit, start_time, depth, next_turn) for child in
                                  children]
                best_val = max(heuristic_opts)
                best_ops_list = [index for index, h_val in enumerate(heuristic_opts) if h_val == best_val]
                best_op = ops[random.choice(best_ops_list)]
                depth += 1
            except TimeoutError:
                return best_op


    def check_double_prob_ops(self, ops):
        if 'pick up' in ops:
            if 'move east' in ops:
                return 'both'
            else:
                return 'pick up'
        if 'move east' in ops:
            return 'move east'
        else:
            return None

    def calc_probability(self, ops):
        # check if move right and pick up package is in the possible ops
        prob_dict = {}
        base_prob = 1 / len(ops)
        answer = self.check_double_prob_ops(ops)
        if answer == 'both':
            x = len(ops)/(len(ops) + 2)
            for op in ops:
                if op == 'pick up':
                    prob_dict[op] = (2 * x) * base_prob
                elif op == 'move east':
                    prob_dict[op] = (2 * x) * base_prob
                else:
                    prob_dict[op] = x * base_prob
        elif answer == 'pick up' or answer == 'move east':
            x = len(ops) / (len(ops) + 1)
            for op in ops:
                if op == 'pick up':
                    prob_dict[op] = (2 * x) * base_prob
                elif op == 'move east':
                    prob_dict[op] = (2 * x) * base_prob
                else:
                    prob_dict[op] = x * base_prob
        else:
            x = len(ops) / len(ops)
            for op in ops:
                prob_dict[op] = x * base_prob

        return prob_dict



    def RB_expectimax(self, env: WarehouseEnv, agent_id, time_limit, start_time, depth, turn):
        now = time.time()
        if now - start_time > time_limit - safety_epsilon:
            raise TimeoutError
        if depth == 0 or env.done():
            return smart_heuristic(env, agent_id)


        ops, children = self.successors(env, agent_id)
        if turn == agent_id:  # max case
            curr_max = -(math.inf)
            for op, child in zip(ops, children):
                v = self.RB_expectimax(child, agent_id, time_limit, start_time, depth - 1, abs(turn - 1))
                curr_max = max(curr_max, v)
            return curr_max

        else:  # probabilistic case
            sum = 0
            probs_dict = self.calc_probability(ops)
            for op, child in zip(ops, children):
                sum += probs_dict[op] * self.RB_expectimax(child, agent_id, time_limit, start_time, depth - 1, abs(turn - 1))
            return sum






# here you can check specific paths to get to know the environment
class AgentHardCoded(Agent):
    def __init__(self):
        self.step = 0
        # specifiy the path you want to check - if a move is illegal - the agent will choose a random move
        self.trajectory = ["move east", "pick up", "move east", "move south", "move south", "move south", "move south",
                           "move east", "drop off", "move south", "move south", "drop off"]

    def run_step(self, env: WarehouseEnv, robot_id, time_limit):
        if self.step == len(self.trajectory):
            return self.run_random_step(env, robot_id, time_limit)
        else:
            op = self.trajectory[self.step]
            if op not in env.get_legal_operators(robot_id):
                op = self.run_random_step(env, robot_id, time_limit)
            self.step += 1
            return op

    def run_random_step(self, env: WarehouseEnv, robot_id, time_limit):
        operators, _ = self.successors(env, robot_id)

        return random.choice(operators)