
from mdp_rl_implementation import value_iteration, get_policy, policy_evaluation, policy_iteration, adp_algorithm
#from mdp import MDP
from mdp import MDP, Action, format_transition_function, print_transition_function
#
from simulator import Simulator
#import numpy as np

def example_driver():
    """
    This is an example of a driver function, after implementing the functions
    in "mdp_rl_implementation.py" you will be able to run this code with no errors.
    """

    mdp = MDP.load_mdp()

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@ The board and rewards @@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    
    mdp.print_rewards()

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@@@@ Value iteration @@@@@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    U = [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]

    print("\nInitial utility:")
    mdp.print_utility(U)
    print("\nFinal utility:")
    U_new = value_iteration(mdp, U)
    mdp.print_utility(U_new)
    print("\nFinal policy:")
    policy = get_policy(mdp, U_new)
    mdp.print_policy(policy)

    
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print("@@@@@@@@@ Policy iteration @@@@@@@@")
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')

    print("\nPolicy evaluation:")
    U_eval = policy_evaluation(mdp, policy)
    mdp.print_utility(U_eval)


    policy = [['UP', 'UP', 'UP', None],
              ['UP', None, 'UP', None],
              ['UP', 'UP', 'UP', 'UP']]


    print("\nInitial policy:")
    mdp.print_policy(policy)
    print("\nFinal policy:")
    policy_new = policy_iteration(mdp, policy)
    mdp.print_policy(policy_new)

    print("Done!\n")
    




def adp_example_driver():
    sim = Simulator()
    reward_matrix, transition_probabilities = adp_algorithm(sim, num_episodes=10)

    print("Reward Matrix:\n")
    print(reward_matrix)

    formatted_transitions = format_transition_function(transition_probabilities)
    print("\nTransition Probabilities:\n")
    print_transition_function(formatted_transitions)


def adp_HW_driver(num_episodes):
    sim = Simulator()
    reward_matrix, transition_probabilities = adp_algorithm(sim, num_episodes=num_episodes)

    print("Reward Matrix:")
    print(reward_matrix)

    formatted_transitions = format_transition_function(transition_probabilities)
    print("\nTransition Probabilities:")
    print_transition_function(formatted_transitions)
    return formatted_transitions


    
if __name__ == '__main__':
    # run our example
    example_driver()
    adp_example_driver()
    print("-------------------------------------------------------tests---------------------------------------------")
    #################################10##############################
    num_episodes = 10
    print(f"results for {num_episodes} episodes:\n")
    transition_probabilities = adp_HW_driver(num_episodes)
    mdp10 = MDP.load_mdp(transition_function="transition_function10")

    policy10 = [['UP', 'UP', 'UP', None],
                ['UP', None, 'UP', None],
                ['UP', 'UP', 'UP', 'UP']]

    print("\nInitial policy:\n")
    mdp10.print_policy(policy10)
    print("\nFinal policy:\n")
    policy_new10 = policy_iteration(mdp10, policy10)
    mdp10.print_policy(policy_new10)

    #################################10##############################
    num_episodes = 100
    print(f"results for {num_episodes} episodes:\n")
    transition_probabilities = adp_HW_driver(num_episodes)
    mdp100 = MDP.load_mdp(transition_function="transition_function100")

    policy100 = [['UP', 'UP', 'UP', None],
                 ['UP', None, 'UP', None],
                 ['UP', 'UP', 'UP', 'UP']]

    print("\nInitial policy:\n")
    mdp100.print_policy(policy100)
    print("\nFinal policy:\n")
    policy_new100 = policy_iteration(mdp100, policy100)
    mdp100.print_policy(policy_new100)

    #################################10##############################
    num_episodes = 1000
    print(f"results for {num_episodes} episodes:\n")
    transition_probabilities = adp_HW_driver(num_episodes)
    mdp1000 = MDP.load_mdp(transition_function="transition_function1000")

    policy1000 = [['UP', 'UP', 'UP', None],
                  ['UP', None, 'UP', None],
                  ['UP', 'UP', 'UP', 'UP']]



    print("\nInitial policy:\n")
    mdp1000.print_policy(policy1000)
    print("\nFinal policy:\n")
    policy_new1000 = policy_iteration(mdp1000, policy1000)
    mdp1000.print_policy(policy_new1000)
