import numpy as np
import matplotlib.pyplot as plt


def initialize_q_function(time_horizon, initial_wealth, risky_return_up, risky_return_down, risk_free_rate,
                          action_options):
    """
    Initialize the state-action value function (Q-function) for Q-learning.

    Parameters:
    - time_horizon (int): The total number of time steps (T).
    - initial_wealth (int): The initial wealth value (W0).
    - risky_return_up (float): The return rate of the risky asset in the "up" state (a).
    - risky_return_down (float): The return rate of the risky asset in the "down" state (b).
    - risk_free_rate (float): The return rate of the risk-free asset (r).
    - action_options (list): A list of possible actions (investment proportions).

    Returns:
    - q_function (list): A list of dictionaries representing the Q-function for each time step.
    """
    q_function = [{} for _ in range(time_horizon)]
    q_function[0] = {initial_wealth: {action: 0 for action in action_options}}

    for step in range(1, time_horizon):
        for current_wealth, actions in q_function[step - 1].items():
            for chosen_action in action_options:
                wealth_up = int(current_wealth + current_wealth * chosen_action * risky_return_up + current_wealth * (
                            1 - chosen_action) * risk_free_rate)
                wealth_down = int(
                    current_wealth + current_wealth * chosen_action * risky_return_down + current_wealth * (
                                1 - chosen_action) * risk_free_rate)

                if wealth_up not in q_function[step]:
                    q_function[step][wealth_up] = {action: 0 for action in action_options}
                if wealth_down not in q_function[step]:
                    q_function[step][wealth_down] = {action: 0 for action in action_options}

    return q_function


def calculate_utility(wealth, risk_aversion=0.0001):
    """
    Calculate the utility of a given wealth level using an exponential utility function.

    The exponential utility function is commonly used to model risk-averse behavior.
    It has the form: U(wealth) = (1 - exp(-α * wealth)) / α, where α is the risk aversion parameter.

    Parameters:
    - wealth (float): The wealth level for which to calculate the utility.
    - risk_aversion (float): The risk aversion parameter (α).
                             Higher values indicate greater risk aversion. Default is 0.0001.

    Returns:
    - utility (float): The utility value corresponding to the given wealth level.
    """
    return (1 - np.exp(-risk_aversion * wealth)) / risk_aversion


def get_max_q_value(q_values):
    """
   Find the maximum Q-value and the corresponding action from a dictionary of Q-values.

   Parameters:
   - q_values (dict): A dictionary where keys are actions and values are Q-values.

   Returns:
   - max_val (float): The maximum Q-value found in the dictionary.
   - best_action (any): The action corresponding to the maximum Q-value.
                       Returns None if the dictionary is empty.
    """
    max_val = -np.inf
    best_action = None
    for act, val in q_values.items():
        if val > max_val:
            max_val = val
            best_action = act
    return max_val, best_action


def compute_average_q_value(q_function, time_step):
    """
    Calculate the average Q-value for a given time step in the Q-function.

    Parameters:
    - q_function (list): A list of dictionaries representing the Q-function for each time step.
    - time_step (int): The specific time step for which to compute the average Q-value.

    Returns:
    - average (float): The average Q-value for the given time step.
                      Returns 0 if no valid Q-values are found.
    """
    total = 0
    count = 0
    for wealth, actions in q_function[time_step].items():
        for act, val in actions.items():
            if val != 0:
                count += 1
                total += val
    return total / count if count > 0 else 0


def train_q_learning(q_function, time_horizon, episodes, learning_rate, discount_factor, exploration_rate, num_actions,
                     action_options, initial_wealth, risky_return_up, risky_return_down, prob_up, risk_free_rate,
                     track_convergence=False):
    """
    Perform Q-learning to update the state-action value function (Q-function).

    Parameters:
    - q_function (list): The initial Q-function, a list of dictionaries for each time step.
    - time_horizon (int): The total number of time steps (T).
    - episodes (int): The total number of training episodes (N).
    - learning_rate (float): The learning rate (α) for updating Q-values.
    - discount_factor (float): The discount factor (γ) for future rewards.
    - exploration_rate (float): The exploration rate (ε) for epsilon-greedy policy.
    - num_actions (int): The number of possible actions.
    - action_options (list): A list of possible actions (investment proportions).
    - initial_wealth (int): The initial wealth value (W0).
    - risky_return_up (float): The return rate of the risky asset in the "up" state (a).
    - risky_return_down (float): The return rate of the risky asset in the "down" state (b).
    - prob_up (float): The probability of the risky asset going "up".
    - risk_free_rate (float): The return rate of the risk-free asset (r).
    - track_convergence (bool): Whether to track the convergence of Q-values. Default is False.

    Returns:
    - q_function (list): The updated Q-function after training.
    - convergence_data (list): A list of lists containing the average Q-values at each time step for each episode.
                               Only returned if track_convergence is True.
    """
    convergence_data = [[] for _ in range(time_horizon)]

    for episode in range(episodes):
        wealth = initial_wealth
        if episode % (episodes / 10) == 0:
            exploration_rate *= 0.7

        for step in range(time_horizon):
            if np.random.rand() < exploration_rate:
                chosen_action = action_options[np.random.randint(0, num_actions)]
            else:
                _, chosen_action = get_max_q_value(q_function[step][wealth])

            if np.random.rand() < prob_up:
                new_wealth = int(
                    wealth + wealth * chosen_action * risky_return_up + wealth * (1 - chosen_action) * risk_free_rate)
            else:
                new_wealth = int(
                    wealth + wealth * chosen_action * risky_return_down + wealth * (1 - chosen_action) * risk_free_rate)

            reward = calculate_utility(new_wealth)

            if step == time_horizon - 1:
                q_function[step][wealth][chosen_action] += learning_rate * (
                            reward - q_function[step][wealth][chosen_action])
                if track_convergence:
                    for t in range(time_horizon):
                        convergence_data[t].append(compute_average_q_value(q_function, t))
                break

            max_q, _ = get_max_q_value(q_function[step + 1][new_wealth])
            q_function[step][wealth][chosen_action] += learning_rate * (
                        reward + discount_factor * max_q - q_function[step][wealth][chosen_action])
            wealth = new_wealth

    if track_convergence:
        return q_function, convergence_data
    else:
        return q_function


# Test the Q-learning algorithm
if __name__ == '__main__':
    # Initialize parameters
    time_horizon = 10
    initial_wealth = 10000

    # Asset parameters
    risky_return_up = 0.05
    risky_return_down = -0.05
    prob_up = 0.7
    risk_free_rate = 0.02

    # Define action space
    num_actions = 10
    action_options = np.linspace(0, 1, num_actions)
    action_options = np.round(action_options, 1)

    # Q-learning parameters
    episodes = 1000000
    learning_rate = 0.01
    discount_factor = 0.1
    exploration_rate = 1

    # Initialize Q-function
    q_function = initialize_q_function(time_horizon, initial_wealth, risky_return_up, risky_return_down, risk_free_rate,
                                       action_options)

    # Train Q-learning
    q_function = train_q_learning(q_function, time_horizon, episodes, learning_rate, discount_factor, exploration_rate,
                                  num_actions, action_options, initial_wealth, risky_return_up, risky_return_down,
                                  prob_up, risk_free_rate)

    for _ in range(5):
        wealth = initial_wealth
        print(f'Trace{_}:')
        for step in range(time_horizon):
            _, action = get_max_q_value(q_function[step][wealth])
            print(f'Time = {step}, Wealth = {wealth}, Action = {action}')
            if np.random.rand() < prob_up:
                wealth = int(wealth + wealth * action * risky_return_up + wealth * (1 - action) * risk_free_rate)
            else:
                wealth = int(wealth + wealth * action * risky_return_down + wealth * (1 - action) * risk_free_rate)
